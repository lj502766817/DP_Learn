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

# 梯度运算,就是用膨胀的结果减去腐蚀的结果,得到的是图形的一个边缘
circular = cv2.imread('./../data/circular.png')
gradient = cv2.morphologyEx(circular, cv2.MORPH_GRADIENT, kernel)
# cv_show('gradient', gradient)

# 顶帽与黑帽操作
img = cv2.imread("./../data/maoci.png")
# 顶帽,原始输入-开运算结果,就是得到了原图像里的毛刺
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv_show('tophat', tophat)
# 黑帽,闭运算-原始输入,就是得到原始图的一个轮廓
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('blackhat', blackhat)
