# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from . import configs

from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # 将q,k,v向量分成12头(16,197,768)->(16,197,12,64)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        print(new_x_shape)
        x = x.view(*new_x_shape)
        print(x.shape)
        # 重新整理矩阵,变成16个样本,每个样本分成12头,每个头的序列长是197,每个序列里的token是64维
        print(x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        print(hidden_states.shape)
        # 构建每个token的query向量
        mixed_query_layer = self.query(hidden_states)
        print(mixed_query_layer.shape)
        # 构建每个token的key向量
        mixed_key_layer = self.key(hidden_states)
        print(mixed_key_layer.shape)
        # 构建每个token的value向量
        mixed_value_layer = self.value(hidden_states)
        print(mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        print(query_layer.shape)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        print(key_layer.shape)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        print(value_layer.shape)

        # 将q和k内积计算权重值,(197,64)*(64,197)得到每个序列与其他197个序列的注意力值
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        print(attention_scores.shape)
        # 削减一下向量维度大小带来的影响
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print(attention_scores.shape)
        attention_probs = self.softmax(attention_scores)
        print(attention_probs.shape)
        # 按权重计算比例
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        print(attention_probs.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        print(context_layer.shape)
        # 重新整理结构(16,12,197,64)->(16,197,12,64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print(context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 将多头的数据合并起来
        context_layer = context_layer.view(*new_context_layer_shape)
        print(context_layer.shape)
        # 通过全连接整合一下特征
        attention_output = self.out(context_layer)
        print(attention_output.shape)
        attention_output = self.proj_dropout(attention_output)
        print(attention_output.shape)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        # 把图像数据做成一个个patch的序列,把一个224*224的图像,切成了多个16*16的patch,并且每个patch都是一个768维的向量
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        # 预置分类任务的cls_token,最后是用这个cls_token去做分类
        # 因为后面会把每个patch做成768维向量,所以这里预置的一个token是(1,1,768),第一个表示是一个数据,第二个表示这是序列里的一个patch,
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # 原始图片数据(16,3,224,224)
        print(x.shape)
        # 一个批次16个数据,先缓存下来
        B = x.shape[0]
        # 因为一个batch是16个数据,所以把cls_token扩展成16,每个数据分一个cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        print(cls_tokens.shape)
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        # 每个样本都被分成了(768,14,14)的特征图,即把原来的图像数据切成了一个个patch
        print(x.shape)
        # 把特征图拉平,这样特征图就变成了一个具有14*14=196个token,每个token都是768维的一个特征序列
        x = x.flatten(2)
        print(x.shape)
        # 把序列长度和token长度换一下(16,196,768),这样就是一个batch里有16个样本,每个样本都是一个长度为196的序列,序列里的每个token都是768维
        x = x.transpose(-1, -2)
        print(x.shape)
        # 把cls_token加上去
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)

        # 加入位置编码的数据
        embeddings = x + self.position_embeddings
        print(embeddings.shape)
        embeddings = self.dropout(embeddings)
        print(embeddings.shape)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        print(x.shape)
        # 将原始数据缓存一下,后面做残差连接用
        h = x
        # 做LN,层标准化
        x = self.attention_norm(x)
        print(x.shape)
        # multi-head Attention
        x, weights = self.attn(x)
        # 进行残差连接
        x = x + h
        print(x.shape)

        h = x  # 再次备份数据,还要做一次残差连接
        x = self.ffn_norm(x)  # 第一次残差连接后的结果再做一次LN
        print(x.shape)
        x = self.ffn(x)  # MLP层,就是一个两层的感知机,主要是把cls_token过两层感知机,最后用来分类用
        print(x.shape)
        x = x + h  # 再过一层残差连接
        print(x.shape)
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            """ 
            linux下路径按照这个
            
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            """
            query_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size,
                                                                                          self.hidden_size).t()
            key_weight = np2th(weights[ROOT + "/" + ATTENTION_K + "/" + "kernel"]).view(self.hidden_size,
                                                                                        self.hidden_size).t()
            value_weight = np2th(weights[ROOT + "/" + ATTENTION_V + "/" + "kernel"]).view(self.hidden_size,
                                                                                          self.hidden_size).t()
            out_weight = np2th(weights[ROOT + "/" + ATTENTION_OUT + "/" + "kernel"]).view(self.hidden_size,
                                                                                          self.hidden_size).t()

            query_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "bias"]).view(-1)
            key_bias = np2th(weights[ROOT + "/" + ATTENTION_K + "/" + "bias"]).view(-1)
            value_bias = np2th(weights[ROOT + "/" + ATTENTION_V + "/" + "bias"]).view(-1)
            out_bias = np2th(weights[ROOT + "/" + ATTENTION_OUT + "/" + "bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + "/" + FC_0 + "/" + "kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + "/" + FC_1 + "/" + "kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + "/" + FC_0 + "/" + "bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + "/" + FC_1 + "/" + "bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/" + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/" + "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/" + "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/" + "bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        print(hidden_states.shape)
        attn_weights = []
        for layer_block in self.layer:
            # 12层self-attention的堆叠
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        # 把多次堆叠的结果最后过一个LN
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # 输入一个图片数据(16,3,224,224),将图片数据去做embedding
        embedding_output = self.embeddings(input_ids)
        # 将已经做成patch的数据,进行encoder
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        # 最后全连接层初始值为0
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        # 最后经过多头之后,最后输出的head层
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        """前向传播"""
        # 先做self-attention
        x, attn_weights = self.transformer(x)
        print(x.shape)
        # 取出cls_token向量,此时cls已经学习到了整个图像的特征,用cls来做分类任务
        logits = self.head(x[:, 0])
        print(logits.shape)

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵计算损失
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
