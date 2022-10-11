"""
文档扫描识别OCR,用opencv做一些形态学的操作进行扫描,然后用tesseract来做扫描后的OCR识别
"""
import numpy as np
import argparse
import cv2
from my_utils import img_resize, cv_show
from PIL import Image
import pytesseract
import os

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    透视变换通常用作图像的矫正,简单说就是把二维的图像先投影到一个三维的空间,然后从三维的空间重新投影到二维,
    这个中间需要一个3*3的转换矩阵来做.原始的二维图像(u1,v1)一般是先加个初始值是1的维度做成三维的(u1,v1,1),然后通过矩阵变换,
    这个矩阵的物理动作就是把原图像做各种平移,旋转等等,线性或非线性的变换,然后得到转换后的图像(u2,v3,w),然后把最后那个不要的维度也
    弄成初始的(u2/w,v2/w,1),然后去掉那个维度就是变换过后的图像了(u3,v3)
    """

    # 获取输入坐标点,将传进来的四个坐标点按顺时针排序
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    # 计算上边和下边的长度取大值
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    # 计算左边和右边的距离取大值
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 变换后对应坐标位置,目的是把目标的四边形转换成一个矩形图像
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # 计算变换矩阵,原始四边形和目标矩形要通过一个三维的M矩阵来转换
    M = cv2.getPerspectiveTransform(rect, dst)
    # 有了转换矩阵直接调函数做变换
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # 返回变换后结果
    return warped


# 读取输入
image = cv2.imread(args["image"])
# 后面要对图片做压缩,这里先记录一下压缩得比例,做透视变化的时候要用这个比例还原成原始图像
ratio = image.shape[0] / 800.0
orig = image.copy()
# 改变图片大小
image = img_resize(orig, height=800)

# 做预处理,改灰度图,做高斯滤波,做边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
# cv_show("img", image)
# cv_show("gray-Edged", np.hstack((gray, edged)))

# 对找到的边缘做轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
# 取面积前五的轮廓,因为很小的一般是在文档里面的
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    # 计算近似轮廓
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 在大小前五的轮廓里面找是方的那个,应该就是文档的主体了
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv_show("Outline", image)

# 透视变换,https://zhuanlan.zhihu.com/p/386137659
# 因为这里只需要文档周围的方框轮廓的四个点就行了,所以就reshape(4, 2),并用ratio把比例还原回去
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
# 展示结果
print("STEP 3: 变换")
# cv_show("Original", img_resize(orig, height=650))
# cv_show("Scanned", img_resize(ref, height=650))

# 使用Tesseract进行OCR识别
# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如 C:\Program Files (x86)\Tesseract-OCR
# tesseract -v进行测试
# tesseract XXX.png result 得到结果
#
# 如果要写在程序里的话 要导包pip install pytesseract
# windows环境下还要修改这个文件,anaconda/lib/site-packges/pytesseract/pytesseract.py
# tesseract_cmd 修改为绝对路径
# 然后加下TESSDATA_PREFIX环境变量,C:\Program Files (x86)\Tesseract-OCR\tessdata

preprocess = 'blur'
image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 滤波或者做二值化都行
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

# 复制一个新的图片来做
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)
