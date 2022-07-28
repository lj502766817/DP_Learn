"""
导出模型的架构图
先导包
pip install pydot
pip install pydotplus
pip install graphviz
去https://graphviz.gitlab.io/download/ 下载安装
"""
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.utils import plot_model
import os

# 预防万一,可以手动加下环境变量
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Graphviz\bin'

plot_model(ResNet50(), to_file='ResNet50_model.png', show_shapes=True)
