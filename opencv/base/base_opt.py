"""
openvc图像的基本操作
"""
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt

# 打开一个图像数据
img = cv2.imread('./../data/cat.jpg')
print(img.shape)


# 用opencv显示图像
def cv_show(name, img_data, wait_time=0):
    # 这里也能创建多个窗口来显示
    cv2.imshow(name, img_data)
    # 图像窗口的等待时间,时间到了会自动关掉,毫秒级,0表示要手动任意键关闭
    # 自动关闭code=-1,手动关闭返回对应按键的code
    code = cv2.waitKey(wait_time)
    print(code)
    cv2.destroyAllWindows()


# cv_show('image', img)

# 单纯读取灰度图,cv2.IMREAD_COLOR：彩色图像;cv2.IMREAD_GRAYSCALE：灰度图像
img = cv2.imread('./../data/cat.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
# cv_show('image', img, 3000)

# 保存处理后的图像
# cv2.imwrite('./../out/gray_cat.jpg', img)

# 实际上读取的图像是用numpy的ndarray来存的,数据是无符号的8位整形
print(type(img))
print(img.dtype)

# 读取视频数据
# cv2.VideoCapture可以直接读视频数据也能捕获摄像头
video_data = cv2.VideoCapture('./../data/test.mp4')
# 检查是否正常打开
if video_data.isOpened():
    open_status, frame = video_data.read()
else:
    open_status = False

while open_status:
    ret, frame = video_data.read()
    if frame is None:
        # 没读到图像帧就跳出
        break
    if ret:
        # 把读到的图像帧转换成灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 用100ms一帧的速度展示灰度图,或者手动退出键直接退出
        cv2.imshow('result', gray)
        if cv2.waitKey(20) & 0xFF == 27:
            break
# 释放资源,关闭窗口
video_data.release()
cv2.destroyAllWindows()

# 因为数据ndarray,那么就可以任意截取数据
img = cv2.imread('./../data/cat.jpg')
new_cat = img[0:50, 0:200]
# cv_show('cat', new_cat)
# 手动的获取通道图,只要R通道的
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
# cv_show('R', cur_img)
# 也能用opencv的函数来拆颜色通道
b, g, r = cv2.split(img)
print(b.shape)
# 能拆也能合
img = cv2.merge((b, g, r))
print(img.shape)

# 对图像做边界填充
# BORDER_REPLICATE：复制法，也就是复制最边缘像素。
# BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
# BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
# BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
# BORDER_CONSTANT：常量法，常数值填充。

# 在周围填充50个像素
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

# plt.show()

# 在ndarray的基础上,可以做任意的数值计算
img_cat = cv2.imread('./../data/cat.jpg')
img_dog = cv2.imread('./../data/dog.jpg')
img_cat2 = img_cat + 10
print(img_cat2[:5, :, 0])
# 直接的+操作会有uint的溢出,结果相当于%256
print((img_cat + img_cat2)[:5, :, 0])
# 想要不溢出,到达最大直接是255的话用add
print(cv2.add(img_cat, img_cat2)[:5, :, 0])

# 两个图像可以用addWeighted来做融合,但是前提是两个图像的shape相同
img_dog = cv2.resize(img_dog, (500, 414))
# 融合后的图像就是0.4的cat+0.6的dog,在加0的偏置
result = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
cv_show('merge', result)

# opencv的resize也可以用比例来做
res = cv2.resize(img, (0, 0), fx=1, fy=3)
cv_show('resize', result)
