import cv2
import numpy as np
from my_utils import cv_show

img = cv2.imread('ocr_a_reference.png')
print('img.shape:', img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print('dst.shape:', dst.shape)

# 把检测到的区域标记成红色
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv_show("img", img)
