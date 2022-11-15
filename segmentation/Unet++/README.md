### UNet++说明

#### UNet网络

UNet网络整体就是一个编码解码的过程.编码就是先将原始图像一层一层的做卷积下采样提取特征,解码就是将最终的特征图再一层一层的做上采样并与同层的特征图拼接,最后还原到原始图像做每个像素点的分类.

![UNet网络](https://user-images.githubusercontent.com/28779173/201855429-bd7d06dc-3d9d-4848-8621-ae748ac9267f.png)

![UNet++网络](https://user-images.githubusercontent.com/28779173/201855472-d8e1e5c6-e9a0-4821-b173-eedcb9089cc8.png)

