import cv2
from my_utils import cv_show

img = cv2.imread('credit_card_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建sift特征提取器,并提取特征
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
# 看看提取到的关键点
img = cv2.drawKeypoints(gray, kp, img)
cv_show("img", img)
# 然后计算特征点的特征值
kp, des = sift.compute(gray, kp)
print(des.shape)
