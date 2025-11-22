import cv2
import numpy as np
import matplotlib.pyplot as plt


def correct_perspective(image_path, layer=1):
    """
    对图像进行透视校正，使其看起来像从正上方拍摄的一样。

    参数：
        image_path (str): 图像路径。
        camera_height (float): 相机离盘子的高度（毫米）。
        camera_angle_deg (float): 相机与水平面的夹角（度）。
        vertical_scaling_factor (float): 纵向拉伸比例调整因子。
        horizontal_scaling_factor (float): 横向拉伸比例调整因子。

    返回：
        corrected_image (numpy.ndarray): 校正后的图像。
    """

    if layer == 2:
        camera_height = 210  # 相机离盘子的高度，单位：毫米
        camera_angle_deg = 50  # 相机与水平面的夹角，单位：度
        vertical_scaling_factor = 0.85  # 纵向调整因子，可以根据需要进行调整
        horizontal_scaling_factor = 0.25  # 横向调整因子，可以根据需要进行调整
    elif layer == 3:
        camera_height = 150  # 相机离盘子的高度，单位：毫米
        camera_angle_deg = 50  # 相机与水平面的夹角，单位：度
        vertical_scaling_factor = 1.15  # 纵向调整因子，可以根据需要进行调整
        horizontal_scaling_factor = 0.24  # 横向调整因子，可以根据需要进行调整
    elif layer == 4:
        camera_height = 90  # 相机离盘子的高度，单位：毫米
        camera_angle_deg = 50  # 相机与水平面的夹角，单位：度
        vertical_scaling_factor = 12  # 纵向调整因子，可以根据需要进行调整
        horizontal_scaling_factor = 0.05  # 横向调整因子，可以根据需要进行调整
    else:
        # 全部当作layer = 1
        camera_height = 270  # 相机离盘子的高度，单位：毫米
        camera_angle_deg = 50  # 相机与水平面的夹角，单位：度
        vertical_scaling_factor = 0.25  # 纵向调整因子，可以根据需要进行调整
        horizontal_scaling_factor = 0.15  # 横向调整因子，可以根据需要进行调整

    # 读取图像
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # 将角度转换为弧度
    theta = np.radians(camera_angle_deg)

    # 计算纵向拉伸比例
    y_indices, x_indices = np.indices((height, width))
    distances_y = (y_indices - height) / np.sin(theta)
    scaling_factors_y = (height - distances_y * np.cos(theta)) / height

    # 对纵向拉伸比例进行调整
    scaling_factors_y = 1 + (scaling_factors_y - 1) * vertical_scaling_factor

    # 对图像进行纵向拉伸
    new_y_indices = np.cumsum(scaling_factors_y[:, 0]).astype(int)
    new_y_indices = new_y_indices - new_y_indices.min()  # 使最小值为0

    new_height = new_y_indices[-1] + 1
    new_image_y = np.zeros((new_height, width, 3), dtype=image.dtype)

    for i in range(height):
        start_y = new_y_indices[i]
        if i < height - 1:
            end_y = new_y_indices[i + 1]
        else:
            end_y = new_height
        new_image_y[start_y:end_y] = image[i]

    # 将结果图像缩放回原图像大小
    resized_image_y = cv2.resize(new_image_y, (width, height), interpolation=cv2.INTER_LINEAR)

    # 计算横向拉伸比例，以图像竖直中线为中心
    half_width = width // 2
    distances_x = np.abs(x_indices - half_width) / np.sin(theta)
    scaling_factors_x = (half_width + distances_x * np.cos(theta)) / half_width

    # 对横向拉伸比例进行调整
    scaling_factors_x = 1 + (scaling_factors_x - 1) * horizontal_scaling_factor

    # 对图像进行横向拉伸
    new_x_indices = np.cumsum(scaling_factors_x[0, :]).astype(int)
    new_x_indices = new_x_indices - new_x_indices.min()  # 使最小值为0

    new_width = new_x_indices[-1] + 1
    new_image_x = np.zeros((height, new_width, 3), dtype=image.dtype)

    for i in range(width):
        start_x = new_x_indices[i]
        if i < width - 1:
            end_x = new_x_indices[i + 1]
        else:
            end_x = new_width
        new_image_x[:, start_x:end_x] = resized_image_y[:, i:i + 1]

    # 将结果图像缩放回原图像大小
    corrected_image = cv2.resize(new_image_x, (width, height), interpolation=cv2.INTER_LINEAR)

    return corrected_image


if __name__ == '__main__':
    # 使用该函数
    image_path = '../egg_tart2_shape.jpg'
    camera_height = 150  # 相机离盘子的高度，单位：毫米
    camera_angle_deg = 50  # 相机与水平面的夹角，单位：度
    vertical_scaling_factor = 0.85  # 纵向调整因子，可以根据需要进行调整
    horizontal_scaling_factor = 0.25  # 横向调整因子，可以根据需要进行调整

    # 需要加上layer，参数改到了函数内部
    corrected_image = correct_perspective(image_path, camera_height, camera_angle_deg, vertical_scaling_factor,
                                          horizontal_scaling_factor)

    # 显示原图和变换后的图像
    original_image = cv2.imread(image_path)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Corrected Image')
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # 保存校正后的图像
    cv2.imwrite('./egg_tart2_angle.jpg', corrected_image)
