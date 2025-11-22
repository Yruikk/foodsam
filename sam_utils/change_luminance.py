import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_gradient_mask(image, light_x=150, light_y=640):

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 创建一个黑到白的水平梯度掩码
    mask = np.zeros((h, w), dtype=np.float32)
    max_distance = np.sqrt(h ** 2 + w ** 2)  # 计算从图像某点到图像边缘的最大距离
    for i in range(h):
        for j in range(w):
            # 根据距离图像某个点（手动设定）的距离来设置掩码值
            distance = np.sqrt((i - light_x) ** 2 + (j - light_y) ** 2)
            mask[i, j] = distance / max_distance

    # 扩展为三通道掩码
    mask = cv2.merge([mask, mask, mask])

    return mask


def apply_gradient_adjustment(image, mask, adjustment_strength=0.5):
    # 将掩码和图像转换为float类型进行运算
    image_float = image.astype(np.float32)
    mask_float = mask.astype(np.float32)

    # 调整掩码的强度
    mask_float = mask_float * adjustment_strength

    # 将掩码应用于图像，调整亮度
    adjusted_image = image_float + (mask_float * 255)

    # 将调整后的图像转换回8位无符号整数类型
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image


def adjust_global_brightness(image, alpha=1.0, beta=0):
    # 将图像转换为float类型进行运算
    image_float = image.astype(np.float32)

    # 调整全局亮度
    adjusted_image = alpha * image_float + beta

    # 将调整后的图像转换回8位无符号整数类型
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

def correct_luminance(image_path, layer=1):
    image = cv2.imread(image_path)
    # 检查图像是否成功读取
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        exit()

    h, w = image.shape[:2]

    if layer == 2:
        light_x = 150
        light_y = w / 2
        alpha = 1.0  # 全局亮度的权重
        beta = -10  # 亮度偏移量
        adjustment_strength = 0.5  # 掩码强度
    elif layer == 3:
        light_x = 200
        light_y = w / 2
        alpha = 1.0  # 全局亮度的权重
        beta = -6  # 亮度偏移量,第二层感觉可以略微beta可以略微调高一点比如-8
        adjustment_strength = 0.40  # 掩码强度
    elif layer == 4:
        light_x = 480
        light_y = w / 2
        alpha = 1.0  # 全局亮度的权重
        beta = -20  # 亮度偏移量,第二层感觉可以略微beta可以略微调高一点比如-8
        adjustment_strength = 0.7  # 掩码强度
    else:
        # 全部当作layer = 1
        light_x = 150
        light_y = w / 2
        alpha = 1.0  # 全局亮度的权重
        beta = -5  # 亮度偏移量
        adjustment_strength = 0.38  # 掩码强度

    gradient_mask = create_gradient_mask(image, light_x, light_y)
    # 应用渐变亮度调整
    image_with_gradient = apply_gradient_adjustment(image, gradient_mask, adjustment_strength=adjustment_strength)

    # 应用全局亮度调整
    final_adjusted_image = adjust_global_brightness(image_with_gradient, alpha=alpha, beta=beta)

    return final_adjusted_image






if __name__ == '__main__':
    #代码重新整合了，不需要这么复杂的调用
    # 读取图像
    image_path = '../egg_tart_shape.jpg'
    save_path = './egg_tart_luminance.jpg'
    image = cv2.imread(image_path)

    # 检查图像是否成功读取
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        exit()

    # 创建渐变掩码
    gradient_mask = create_gradient_mask(image)

    # 调整参数
    alpha = 1.0  # 全局亮度的权重
    beta = -10  # 亮度偏移量
    # adjustment_strength = 0.25  # 掩码强度
    adjustment_strength = 0.5  # 掩码强度

    # 应用渐变亮度调整
    image_with_gradient = apply_gradient_adjustment(image, gradient_mask, adjustment_strength=adjustment_strength)

    # 应用全局亮度调整
    final_adjusted_image = adjust_global_brightness(image_with_gradient, alpha=alpha, beta=beta)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Brightness Adjusted Image')
    plt.imshow(cv2.cvtColor(final_adjusted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # 保存结果
    cv2.imwrite(save_path, final_adjusted_image)
    print(f"Adjusted image saved at {save_path}")
