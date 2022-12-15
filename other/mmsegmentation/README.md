##### 汽车数据集的语义分割

数据集:
https://www.kaggle.com/intelecai/car-segmentation

格式:
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val


启动参数:
../configs/deeplabv3plus_r18-d8_769x769_80k_car-segmentation.py

预训练模型:
https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth
