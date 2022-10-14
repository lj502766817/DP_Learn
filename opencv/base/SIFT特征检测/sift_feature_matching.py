"""
Brute-Force蛮力匹配:1对1的匹配,k对最佳匹配
"""
import cv2
import numpy as np
from my_utils import cv_show

# 读取两个图片
img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box_in_scene.png', 0)
# cv_show("img1", img1)
# cv_show("img2", img2)

# 创建sift特征提取器
sift = cv2.xfeatures2d.SIFT_create()
# 提取两个图片上的特征
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 直接一个特征点一个特征点的去暴力匹配
# crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
# NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(des1, des2)
# 按特征匹配的距离做排序
matches = sorted(matches, key=lambda x: x.distance)
# 选出前15的特征点
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], None, flags=2)
cv_show("img3", img3)

# k对最佳匹配,就是把样本图像里的关键点与训练图像里的关键点做多个匹配,只有k对匹配成功了,才算成功
# 例如,用样本图像的苹果去训练图像里匹配,如果匹配到两个绿苹果就认为样本图像是绿苹果,如果一个是红,一个是绿就匹配失败
bf = cv2.BFMatcher()
# 一个match里是按距离排序好的k个结果
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    # 如果两个匹配好的特征点够接近的话,就选出来画图
    if m.distance < 0.5 * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('img3', img3)

# 一个更快速的matcher,用法一样,只不过这个快些
fb = cv2.FlannBasedMatcher()
matches = fb.match(des1, des2)
# 按特征匹配的距离做排序
matches = sorted(matches, key=lambda x: x.distance)
# 选出前15的特征点
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], None, flags=2)
cv_show("img3", img3)
