import cv2
import numpy as np


def get_homography(src_pts, dst_pts):
    """
    计算从源点到目标点的透视变换矩阵。

    参数:
    src_pts (numpy.ndarray): 源点数组，形状为 (4, 2)
    dst_pts (numpy.ndarray): 目标点数组，形状为 (4, 2)

    返回:
    numpy.ndarray: 透视变换矩阵，形状为 (3, 3)
    """
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)
    return H


def get_source_points(w, h, layer=1):
    """
    定义源点（梯形的四个点）。

    参数:
    w (int): 图像的宽度
    h (int): 图像的高度

    返回:
    list: 源点列表，包含 4 个 (x, y) 元组
    """
    if layer == 2:
        x1, y1 = 250, 250  # 点 1 (x1, y1)
        x2, y2 = w - 150, 250  # 点 2 (x2, y2)
        x3, y3 = w, h - 20  # 点 3 (x3, y3)
        x4, y4 = 0, h  # 点 4 (x4, y4)
    elif layer == 3:
        x1, y1 = 200, 165  # 点 1 (x1, y1)
        x2, y2 = w - 110, 170  # 点 2 (x2, y2)
        x3, y3 = w, h - 110  # 点 3 (x3, y3)
        x4, y4 = 0, h - 120  # 点 4 (x4, y4)
    elif layer == 4:
        x1, y1 = 195, 70  # 点 1 (x1, y1)
        x2, y2 = w - 130, 80  # 点 2 (x2, y2)
        x3, y3 = w, h - 220  # 点 3 (x3, y3)
        x4, y4 = 0, h - 250  # 点 4 (x4, y4)
    else:
        # 全部当作layer = 1
        x1, y1 = 300, 310  # 点 1 (x1, y1)
        x2, y2 = w - 220, 320  # 点 2 (x2, y2)
        x3, y3 = w, h  # 点 3 (x3, y3)
        x4, y4 = 40, h  # 点 4 (x4, y4)


    src_points = [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ]

    return src_points


def warp_image(image_path, layer=1):
    """
    加载图像并应用透视变换。

    参数:
    image_path (str): 图像的路径

    返回:
    tuple: 原始图像和变换后的图像
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None

    clone = image.copy()

    # 获取图像尺寸
    h, w = clone.shape[:2]

    # 获取源点
    src_points = get_source_points(w, h, layer)

    # 目标点（矩形的四个角）
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    # 将源点转换为 numpy 数组
    src_pts = np.array(src_points, dtype="float32")

    # 计算透视变换矩阵并应用
    H = get_homography(src_pts, dst_pts)
    warped = cv2.warpPerspective(clone, H, (w, h))

    return image, warped

# def warp_image(image, layer=1):
#     """
#     加载图像并应用透视变换。
#
#     参数:
#     image_path (str): 图像的路径
#
#     返回:
#     tuple: 原始图像和变换后的图像
#     """
#
#     clone = image.copy()
#
#     # 获取图像尺寸
#     h, w = clone.shape[:2]
#
#     # 获取源点
#     src_points = get_source_points(w, h, layer)
#
#     # 目标点（矩形的四个角）
#     dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
#
#     # 将源点转换为 numpy 数组
#     src_pts = np.array(src_points, dtype="float32")
#
#     # 计算透视变换矩阵并应用
#     H = get_homography(src_pts, dst_pts)
#     warped = cv2.warpPerspective(clone, H, (w, h))
#
#     return image, warped


def display_images(original, warped):
    """
    显示原始图像和变换后的图像。

    参数:
    original (numpy.ndarray): 原始图像
    warped (numpy.ndarray): 变换后的图像
    """
    cv2.imshow("Original Image", original)
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, output_path):
    """
    保存图像到指定路径。

    参数:
    image (numpy.ndarray): 需要保存的图像
    output_path (str): 图像的保存路径
    """
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    # 图像的路径
    image_path = '../egg_tart2.jpg'
    # 保存变换后图像的路径
    output_path = '../egg_tart2_shape.jpg'

    # 使用提供的源点进行图像变换
    # 需要加上layer
    original, warped = warp_image(image_path)

    # 如果变换成功，显示图像并保存变换后的图像
    if original is not None and warped is not None:
        print('in')
        # display_images(original, warped)
        save_image(warped, output_path)
        print(f"Warped image saved at {output_path}")
