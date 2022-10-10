"""
用OpenCV的形态学操作和模板匹配进行信用卡号的OCR识别
"""
from imutils import contours
import numpy as np
import argparse
import cv2
from my_utils import sort_contours, img_resize, cv_show

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img', img)
# 转灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)
# 把灰度图做成二值化的图像,因为这个模板就是黑白的,所以阈值就直接用10
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓,这里只需要refCnts轮廓的结果就行了,原图和层次关系都用不上
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 展示轮廓
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)
print(np.array(refCnts).shape)
# 排序，从左到右，从上到下
refCnts = sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓,把模板里的数字一个个拆出来
for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每一个数字对应每一个小模板图像,后续会用信用卡里拆出来的数字轮廓去一个个的做比对
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image', image)
# 固定图像的大小
image = img_resize(image, width=300)
# 同样转成灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作,原始输入-开运算(就是先做腐蚀操作,然后再做膨胀操作)，突出灰度中更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)
# ksize=-1相当于用3*3的
# 通过Sobel算子把轮廓把边缘轮廓找出来,这里就找的x方向,效果更好点
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                  ksize=-1)

# 先做个绝对值,再做个归一化,可以直接用gradX = cv2.convertScaleAbs(gradX, alpha=0.5)代替,不过要注意参数alpha的选择
# gradX = cv2.convertScaleAbs(gradX, alpha=0.5)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字区域连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh1', thresh)

# 再来一个闭操作,使数字区域更好的连接到一个整体
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh2', thresh)

# 计算轮廓
thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算每个轮廓的外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 根据长宽比和数字区域的长宽范围,来筛选合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if (2.5 < ar < 4.0) and (40 < w < 55) and (10 < h < 20):
        # 符合的留下来
        locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 找到数字区域
rectangle_img = image.copy()
for (x, y, w, h) in locs:
    rectangle_img = cv2.rectangle(rectangle_img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
cv_show("rectangle_img", rectangle_img)

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):

    groupOutput = []

    # 在灰度图中,根据坐标提取每一个数字组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group1', group)
    # 预处理,做成二值化
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group2', group)
    # 计算每一组的轮廓
    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    # 把每一组的轮廓从左到右排序
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的外接矩形轮廓
        (x, y, w, h) = cv2.boundingRect(c)
        # 从二值化的图中,把这个数字扣出来
        roi = group[y:y + h, x:x + w]
        # 把大小做成和前面模板里的一样
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分,这里就相当于拿卡里拆出来的数字当模板去模板图片里一个个的匹配
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来展示下
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
