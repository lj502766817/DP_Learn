"""
交通速度预测
"""
from torch_geometric_temporal.dataset import METRLADatasetLoader
import seaborn as sns
import matplotlib.pyplot as plt

loader = METRLADatasetLoader()
# 输入12个样本点 预测未来12个点的值
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("样本个数: ", dataset.snapshot_count)
# 一个图里207个点(传感器),每个点是有12个时间点的速度值(两个特征:速度,时间),标签是12个时间点的速度值
print(next(iter(dataset)))

# 看看数据分布,1号传感器的24小时记录的速度值
sensor_number = 1
hours = 24
sensor_labels = [bucket.y[sensor_number][0].item() for bucket in list(dataset)[:hours]]
sns.lineplot(data=sensor_labels)
plt.show()
