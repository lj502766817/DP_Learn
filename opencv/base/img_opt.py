"""
图像处理操作
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img_data, wait_time=0):
    cv2.imshow(name, img_data)
    code = cv2.waitKey(wait_time)
    cv2.destroyAllWindows()


# 图像的阈值处理,根据灰度图的每个像素点值的大小,对每个像素点进行处理
# cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0
# cv2.THRESH_BINARY_INV THRESH_BINARY的反转
# cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
# cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
# cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转

img = cv2.imread("./../data/cat.jpg")
img_gray = cv2.imread("./../data/cat.jpg", cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

# 图像的平滑处理,对一些有噪声点的图片进行处理,能将噪声点变得平滑
img = cv2.imread("./../data/lenaNoise.png")
# cv_show('img', img)
# 均值滤波,按照设定的卷积核大小,对每个像素点区域做平均卷积操作
blur = cv2.blur(img, (3, 3))
# cv_show('blur', blur)
# 方框滤波,基本上和均值滤波操作一样,只是多了些设置选择,并且可以选择是否做归一化操作,不做归一化的话容易使像素值越界
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# cv_show('box', box)
# 高斯滤波,高斯滤波的卷积核是根据中心点和周围点的差别,通过高斯分布来计算卷积核的值,这样就更重视中心点了
gaussian = cv2.GaussianBlur(img, (5, 5), 1)
# cv_show('gaussian', gaussian)
# 中值滤波,就是直接用中值来代替了
median = cv2.medianBlur(img, 5)
# cv_show('median', median)
# 全部的对比图
res = np.hstack((img, blur, box, gaussian, median))
# cv_show('res', res)

# 图像的一些形态学操作

# 腐蚀操作,就是根据设定核大小,对图像的边缘进行腐蚀,可以去掉灰度图中的一些毛刺
img = cv2.imread("./../data/maoci.png")
# cv_show('img', img)
# 设置腐蚀核的大小
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# 但是因为是腐蚀操作,所以毛刺虽然去掉了,但是图像本身也变瘦了
# cv_show('erosion', erosion)
# 腐蚀操作是可以进行迭代多次的
circular = cv2.imread('./../data/circular.png')
kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(circular, kernel, iterations=1)
erosion_2 = cv2.erode(circular, kernel, iterations=2)
erosion_3 = cv2.erode(circular, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
# cv_show('res', res)

# 膨胀操作,就是腐蚀操作的反向操作
img = cv2.imread("./../data/maoci.png")
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(img, kernel, iterations=1)
# 就是把图像变粗
# cv_show('dilate', dilate)
# 膨胀同样可以迭代
circular = cv2.imread('./../data/circular.png')
dilate_1 = cv2.dilate(circular, kernel, iterations=1)
dilate_2 = cv2.dilate(circular, kernel, iterations=2)
dilate_3 = cv2.dilate(circular, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
# cv_show('res', res)

# 开运算,就是现做腐蚀操作,然后再做膨胀操作,就是先消掉毛刺,再把变瘦的图像还原回去
img = cv2.imread("./../data/maoci.png")
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv_show('opening', opening)
# 闭运算,跟开运算相反,先膨胀,再腐蚀,等于是把毛刺做的更清晰
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# cv_show('closing', closing)
res = np.hstack((img, opening, closing))
# cv_show('res', res)

# 梯度操作,就是用膨胀的结果减去腐蚀的结果,得到的是图形的一个边缘
circular = cv2.imread('./../data/circular.png')
gradient = cv2.morphologyEx(circular, cv2.MORPH_GRADIENT, kernel)
# cv_show('gradient', gradient)

# 顶帽与黑帽操作
img = cv2.imread("./../data/maoci.png")
# 顶帽,原始输入-开运算结果,就是得到了原图像里的毛刺
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# cv_show('tophat', tophat)
# 黑帽,闭运算-原始输入,就是得到原始图的一个轮廓
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# cv_show('blackhat', blackhat)

# 图像的梯度计算

# soble算子的方式
# 在X方向上为: -1 0 +1 这样的矩阵,在Y方向上为: -1 -2 -1 这样的矩阵
#            -2 0 +2                      0  0  0
#            -1 0 +1                     +1 +2 +1
img = cv2.imread('./../data/circular.png', cv2.IMREAD_GRAYSCALE)
# 这里看是X方向的梯度,算子的大小是3*3的
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# 因为灰度图,白到黑是正数,黑到白就是负数了,在opencv里所有的负数都会被截成0,所以这里就是个半圆弧,因此,梯度的计算需要取绝对值
# cv_show('sobelx', sobelx)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
# cv_show('sobelx', sobelx)
# 分别计算X方向和Y方向,再求和就能得到这个图像的全部梯度值了
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
# cv_show("sobely", sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy', sobelxy)
# 也可以直接在X和Y方向上计算,但是这样效果没有分开计算好
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
# cv_show('sobelxy', sobelxy)

# Scharr算子,格式和soble算子是一样的,但是数值上变大了,这样,使得梯度的计算变得更加敏感
# laplacian算子,laplacian算子跟Scharr与soble不同的点,是它是基于二阶导的,所以它更加敏感,通常是跟一些其他的操作一起做
# 并且laplacian算子没有方向之分了: 0  1  0
#                              1 -4  1
#                              0  1  0
img = cv2.imread('./../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
# 对比查看效果
res = np.hstack((sobelxy, scharrxy, laplacian))
cv_show('res', res)

# Canny边缘检测
# Canny边缘检测是将前面的一些操作综合起来,来做边缘检测
# 1.使用高斯滤波器，以平滑图像，滤除噪声。
# 2.计算图像中每个像素点的梯度强度和方向。
# 3.应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
# 非极大值抑制有两种方法,线性插值法:沿着像素点的梯度方向做延伸,这时,延伸出去的方向在周围像素点围成的方型里会有两个交点,
# 这时就能用线性的方式计算,通过交点两边的的像素点的梯度得到交点的梯度,这样就如果主像素点的梯度小于交点的梯度,就能舍弃掉了.
# 还有一种就是,由于一个像素点周围是被8个像素点包围的,那么可以把这个像素点的梯度离散成这8个方向就能比较了,不用插值了
# 4.应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
# 如果梯度值>maxValue的话,就认为是边界,如果梯度值<minValue的话,就直接舍弃,如果在中间的话,就看这个像素点有没有和边界的像素点连一起,没连上就丢弃
# 5.通过抑制孤立的弱边缘最终完成边缘检测。
img = cv2.imread("./../data/lena.jpg", cv2.IMREAD_GRAYSCALE)
# 不同的阈值对比查看
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)
# 可以看到阈值设置的越窄,对边界就越敏感
res = np.hstack((v1, v2))
cv_show('res', res)
