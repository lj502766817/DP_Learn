import numpy as np
import cv2


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def matchKeypoints(kps_a, kps_b, features_a, features_b, ratio, reproj_thresh):
    # 建立暴力匹配器
    matcher = cv2.BFMatcher()

    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    raw_matches = matcher.knnMatch(features_a, features_b, 2)

    matches = []
    for m in raw_matches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算视角变换矩阵,至少要能找到4个点才能计算H(3,3)矩阵,8个未知数至少8个方程
    if len(matches) > 4:
        # 获取根据匹配的特征的索引,获得对应的图像上的关键点
        pts_a = np.float32([kps_a[i] for (_, i) in matches])
        pts_b = np.float32([kps_b[i] for (i, _) in matches])

        # 计算视角变换矩阵,这里是合在一起做的,
        # 先做RANSAC(随机抽样一致算法,https://blog.csdn.net/zhoucoolqi/article/details/105497572)在筛选出的特征点中找比较好的
        # 然后根据选出来的比较好的特征点点去计算H矩阵以便做投影变换用(看看文档的OCR里的说明)
        # status表示RANSAC筛选后的特征点是否可用的状态
        (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thresh)

        # 返回结果
        return matches, H, status

    # 如果匹配对小于4时，返回None
    return None


def detectAndDescribe(image):
    # 将彩色图片转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 建立SIFT生成器
    descriptor = cv2.xfeatures2d.SIFT_create()
    # 检测SIFT特征点，并计算描述子
    (kps, features) = descriptor.detectAndCompute(image, None)

    # 将特征点结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    # 返回特征点集，及对应的描述特征
    return kps, features


def drawMatches(image_a, image_b, kps_a, kps_b, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = image_a.shape[:2]
    (hB, wB) = image_b.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = image_a
    vis[0:hB, wA:] = image_b

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            pt_a = (int(kps_a[queryIdx][0]), int(kps_a[queryIdx][1]))
            pt_b = (int(kps_b[trainIdx][0]) + wA, int(kps_b[trainIdx][1]))
            cv2.line(vis, pt_a, pt_b, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


class Stitcher:

    # 拼接函数
    def stitch(self, images, ratio=0.75, reproj_thresh=4.0, show_matches=False):
        # 获取输入图片
        (image_left, image_right) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kps_right, features_right) = detectAndDescribe(image_right)
        (kps_left, features_left) = detectAndDescribe(image_left)

        # 匹配两张图片的所有特征点，返回匹配结果
        match_result = matchKeypoints(kps_right, kps_left, features_right, features_left, ratio, reproj_thresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if match_result is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵,并且是右图片往左图片的投影矩阵
        (matches, H, status) = match_result
        # 将右图片进行视角变换，result是变换后图片
        result = cv2.warpPerspective(image_right, H, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))
        cv_show('image_right', image_right)
        cv_show("result_right", result)
        # 将左图片传入result图片最左端
        result[0:image_left.shape[0], 0:image_left.shape[1]] = image_left
        cv_show('result_left+right', result)
        # 检测是否需要显示图片匹配
        if show_matches:
            # 生成匹配图片
            vis = drawMatches(image_right, image_left, kps_right, kps_left, matches, status)
            # 返回结果
            return result, vis

        # 返回匹配结果
        return result
