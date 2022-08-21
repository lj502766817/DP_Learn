import io
import json
import flask
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torch.autograd import Variable

# 初始化Flask app
app = flask.Flask(__name__)
model = None
use_gpu = torch.cuda.is_available()
# 读取标签实际对应的名字
with open('./data/flower_data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# 加载模型
def load_model():
    global model
    # 按之前训练模型那样,用resnet50
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    # 分类是102分类
    model.fc = nn.Linear(num_ftrs, 102)

    # print(model)
    checkpoint = torch.load('./data/flower_data/best.pt')
    model.load_state_dict(checkpoint['state_dict'])
    # 将模型指定为验证模式
    model.eval()
    # 是否使用gpu
    if use_gpu:
        model.cuda()


# 数据预处理
def prepare_image(image):
    # 因为是用RGB格式的训练的,把格式都转成RGB的
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # 把图片修改成模型对应的输入尺寸
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)

    # （RGB三通道归一化）这里的参数和数据集中是对应的，训练过程中一致
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # 增加一个维度，用于按batch测试,本次这里一次测试一张
    image = image[None]
    if use_gpu:
        image = image.cuda()
    # 不需要求导
    return Variable(image, volatile=True)


# 开启服务,这里的predict只是一个名字，可自定义
@app.route("/api/image_recognition_classification", methods=["POST"])
def predict():
    # 做一个标志，刚开始无图像传入时为false，传入图像时为true
    data = {"success": False}

    # 如果收到请求
    if flask.request.method == 'POST':
        # 看入参有没有image
        if flask.request.files.get("image"):
            # 将收到的图像进行读取
            image = flask.request.files["image"].read()
            # 二进制数据
            image = Image.open(io.BytesIO(image))

            # 利用上面的预处理函数将读入的图像进行预处理
            image = prepare_image(image)

            # 使用softmax函数看每个类别的概率值
            preds = F.softmax(model(image), dim=1)
            # 看下概率值前三的
            results = torch.topk(preds.cpu().data, k=3, dim=1)
            # 把值从tensor转成ndarray
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())

            # 将data字典增加一个key,value,其中value为list格式
            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                # label_name = idx2label[str(label)]
                r = {"index": str(label), "label": cat_to_name[str(label)], "probability": float(prob)}
                # 将预测结果添加至data字典
                data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True
    # 将最终结果以json格式文件传出
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    # 先加载模型
    load_model()
    # 再开启服务
    app.run(port='9100')
