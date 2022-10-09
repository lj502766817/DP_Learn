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

# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
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
# cv_show('res', res)

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
# cv_show('res', res)

# 图像金字塔:就是图像的上采样和下采样,下采样就是把一个400*400的图像缩小成200*200的,上采样类似
# 高斯金字塔:下采样,就是将图像和高斯卷积核相乘然后去掉所有的偶数行列;上采样,就是先隔行插入初始值0,把图像扩充一倍,然后再用高斯卷积核去乘,获得近似值
img = cv2.imread("./../data/AM.png")
# cv_show('img', img)
print(img.shape)
# 上采样
up = cv2.pyrUp(img)
# cv_show('up', up)
print(up.shape)
# 下采样
down = cv2.pyrDown(img)
# cv_show('down', down)
print(down.shape)
# 上采样和下采样多图像本身来说是一种损失
up = cv2.pyrUp(img)
up_down = cv2.pyrDown(up)
# cv_show('up_down', up_down)
# 先上采样再下采样也是只是还原了大小,画质上可以看到是损失了
# cv_show('up_down', np.hstack((img, up_down)))

# 还有一种是拉普拉斯金字塔,是先将图像经过下采样然后再做上采样,再用原图减去处理过的图形,这就是一层拉普拉斯采样,多个就是这样做多层
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - down_up
# cv_show('l_1', l_1)

# 图像的轮廓
# mode:轮廓检索模式
# RETR_EXTERNAL ：只检索最外面的轮廓；
# RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
# RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
# RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;(一般用这个,自己手动筛想要的轮廓)
# method:轮廓逼近方法
# CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
# CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。

# 使用二值图像,可以获得更高的准确率
img = cv2.imread('./../data/contours.png')
# 先读灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 然后把灰度图通过阈值操作转换成二值的图像
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv_show('thresh', thresh)
# 轮廓检测函数的第一个结果就是我们的输入的二值图像,第二个结果是检测到的轮廓点,第三个结果是轮廓的层级,因为我们的模式是全部层级都输出的
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# 绘制轮廓,传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy,这个轮廓的绘制操作是直接在输入上做的
draw_img = img.copy()
# -1 表示绘制全部的轮廓
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)

# 轮廓的一些特征也能直接得到
# 先取第一层轮廓
cnt = contours[0]
# 看看面积
print(cv2.contourArea(cnt))
# 看看周长，True表示闭合的
print(cv2.arcLength(cnt, True))

# 轮廓的近似处理
# 整体的原理就是,如果要近似处理一个轮廓,就先在这个轮廓的边的两点作一个直线,然后取轮廓上离这个线最远的那个点算离直线的距离,看是不是小于阈值,
# 小于阈值的话,就是这个直线,如果大于的话,就从这个点往轮廓两边做两个直线,然后对这个两个直线继续做上面的操作
# 先读图像
img = cv2.imread('./../data/contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
# 绘制轮廓
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
# 可以看到这里的轮廓是完全贴着边的
# cv_show('res', res)
# 先设置近似处理的轮廓的差别值
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
# 这里可以看到绘制的轮廓变得粗略了
# cv_show('res', res)

# 轮廓的外接矩形和外接圆
img = cv2.imread('./../data/contours.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
# 返回的外接矩形的四个点的参数
x, y, w, h = cv2.boundingRect(cnt)
# 在图上把外接矩形画上去
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv_show('img', img)
# 这个矩形就是为了得到更多的信息
area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = float(area) / rect_area
print('轮廓面积与边界矩形比', extent)
# 外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
# cv_show('img', img)

# 图像的傅里叶变换,将图像的数据转换到频域上进行处理
# 傅里叶变换的作用
# 高频：变化剧烈的灰度分量，例如边界
# 低频：变化缓慢的灰度分量，例如一片大海
# 滤波
# 低通滤波器：只保留低频，会使得图像模糊
# 高通滤波器：只保留高频，会使得图像细节增强

# 先读进来灰度图
img = cv2.imread('./../data/lena.jpg', 0)
# 做傅里叶变换需要把数据转成float32
img_float32 = np.float32(img)
# 做傅里叶变换
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# 通过把频率为0的部分移到中心来
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的形式,取傅里叶变换后的实部和虚部做下公式的转换
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# 输出原始的灰度图和经过傅里叶变换后的频域的灰度图
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# 傅里叶的低通滤波,就是将频域图像上的高频部分保留下来,低频部分丢掉
img = cv2.imread('./../data/lena.jpg', 0)
img_float32 = np.float32(img)
# 做傅里叶变换并做偏移
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = img.shape
# 得到图像的中心位置
crow, ccol = int(rows / 2), int(cols / 2)
# 做低通滤波,就是做一个mask,需要留下的位子就把设值为1,丢弃的部分就设值为0
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
fshift = dft_shift * mask
# 再把频域的图像做反解析
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# 画图对比,可以看到做了低通滤波后的图像,变得模糊了
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
# plt.show()

# 高通滤波,就是保留下频域中高频的部分,保留下来的就是图像变化分明的边界
img = cv2.imread('./../data/lena.jpg', 0)
img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 中心位置
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
fshift = dft_shift * mask
# IDFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
# 可以看到高通滤波是保留了图像的轮廓信息
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_back, cmap='gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
# plt.show()

# 图像的直方图信息,就是将图像中的像素值信息,通过直方图的形式反应出来
# images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]
# channels: 同样用中括号括来它会告函数我们统幅图像的直方图,如果入图像是灰度图它的值就是[0]如果是彩色图像的话传入的参数可以是[0][1][2]它们分别对应着BGR。
# mask: 掩模图像,整幅图像的直方图就把它设置成None。但是如果你想统图像某一分的直方图的你就制作一个掩模图像并使用它。
# histSize:BIN 的数目,也应用中括号括来
# ranges: 像素值范围常为 [0,256]
img = cv2.imread('./../data/cat.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist.shape)
# 用matplotlib也能展示
# plt.hist(img.ravel(), 256)
# plt.show()
# 画彩色图的
img = cv2.imread('./../data/cat.jpg')
color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()
# 用mask可以只看图像上一部分的直方图数据
# 创建mast,就是一个黑白方框
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255
# cv_show('mask', mask)
img = cv2.imread('./../data/cat.jpg', 0)
# cv_show('img', img)
# 通过与操作来做遮罩,把方框外面的全弄成黑色
masked_img = cv2.bitwise_and(img, img, mask=mask)
# cv_show('masked_img', masked_img)
# 获取直方图信息
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
# 画图展示
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

# 直方图的均衡化,可以使的图像的亮度和色彩做提升
# 原理上就是先对灰度图的各个灰度值进行统计,然后算各个灰度值的概率,然后将各个灰度值的概率转成一个累计概率,再通过累计概率来计算转换后的灰度值
# 加入原来图像上灰度值是50的像素个数是4,然后概率是0.25,那么他是第一个的情况下,他的累计概率就也是0.25,然后计算映射后的灰度值就是0.25*(255-0)=64
img = cv2.imread('./../data/dog.jpg', 0)
# 做直方图均衡化
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
# 可以看到在使灰度图的色彩更加鲜明了
# cv_show('res', res)
# 直方图的均衡化也可以用自适应的方式来做
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
# cv_show('res', res)

# 模板匹配,就是按模板窗口在图像上进行滑动,看哪一块的概率大
# 先把图像和模板读进来
img = cv2.imread('./../data/lena.jpg', 0)
template = cv2.imread('./../data/face.jpg', 0)
# 模板的长宽
h, w = template.shape[:2]
# 相似度的计算有很多种方式,一般用带归一化的效果会好些
# TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
# TM_CCORR：计算相关性，计算出来的值越大，越相关
# TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
# TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
# TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
# TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
# 公式：https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
# 做模板匹配
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
# 根据相似度计算的方式不同,一般取值最大的或者最小的,这个函数的结果就是对应的值以及左上坐标位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# 各种方法的对比
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值,传到函数里的不是字符串,二十对应的方法
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    # plt.subplot(122), plt.imshow(img2, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()

# 也能做多个目标的检测
img_rgb = cv2.imread('./../data/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./../data/mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
