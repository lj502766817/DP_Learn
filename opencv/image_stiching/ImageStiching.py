"""
把图片拼接成全景图,将左图片拼到右图片上
"""
from Stitcher import Stitcher
from Stitcher import cv_show
import cv2

# 读取拼接图片
image_left = cv2.imread("left_01.png")
image_right = cv2.imread("right_01.png")

stitcher = Stitcher()
(result, vis) = stitcher.stitch([image_left, image_right], show_matches=True)

# 显示所有图片
cv_show("Keypoint Matches", vis)
cv_show("Result", result)
