import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        print(scores.shape)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        print(V.shape)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # 找出25个最有价值的Q,算attention,然后返回attention和Q的index
        # Q [B, H, L, D],Q,K都是(32,8,96,64)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # 先增加一个维度，相当于复制，再扩充
        print(K_expand.shape)  # 将K扩展成(32, 8, 96, 96, 64)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q 构建96*25,值在[0,96)的随机数
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        print(K_sample.shape)  # (32, 8, 96, 25, 64) 这里就是对96个Q每个Q都随机选了25个K
        # (32, 8, 96, 1, 64)*(32, 8, 96, 64, 25)计算出了96个Q和25个K之间的关系
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        print(Q_K_sample.shape)  # (32, 8, 96, 25)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # 96个Q中每一个选跟其他K关系最大的值 再计算与均匀分布(25个的平均值)的差异
        print(Q_K_sample.max(-1)[0].shape)
        print(M.shape)
        M_top = M.topk(n_top, sorted=False)[1]  # 对96个Q的评分中选出25个 返回值1表示要得到索引
        print(M_top.shape)  # 最有价值的25个Q的索引

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # 根据索引重构Q
        print(Q_reduce.shape)  # (32, 8, 25, 64)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k 25个Q和全部K之间的关系
        print(Q_K.shape)
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # 算出全部V的均值
            print(V_sum.shape)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # 先把96个V都用均值来替换
            print(contex.shape)
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
            print(contex.shape)
        return contex  # (32, 8, 96, 64),现在这里面的值都是初始化的均值

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            print(scores.shape)
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        print(attn.shape)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 对25个有Q的更新V，其余的没变还是均值
        print(context_in.shape)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            print(attns.shape)
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # 转换形状(批次数,头数,序列长度,特征数)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        # 随机选多少个K向量来评估出有价值的Q
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) Key里要选的个数,当做是个先验来看就行
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q) 需要多少个有价值Q向量

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # 减少维度大小带来的影响
        # get the context,把self-attention先用全局的V平均值做初始
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries,再用那25个有价值的attention重新计算特征替换上去
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        # 最后的结果就是,attention里那25个有价值的attention的V的结果是参与更新的,其他都是V的均值
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads  # 前面初始化的时候是默认做8头
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # 根据输入得到多头后的Q向量(32,96,8,64)
        keys = self.key_projection(keys).view(B, S, H, -1)  # 得到K向量
        values = self.value_projection(values).view(B, S, H, -1)  # 得到V向量

        out, attn = self.inner_attention(  # 重头戏,计算ProbAttention
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        # 最后把全部特征值特征整合一下后输出
        return self.out_projection(out), attn
