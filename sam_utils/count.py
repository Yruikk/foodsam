import numpy as np
import cv2
import matplotlib.pyplot as plt


def count_area_num(bool_input, img_title, kernel_size=47):
    gray_matrix = np.array(bool_input, dtype=np.uint8) * 255
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * 5 + 1, 2 * 5 + 1), (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
    dilated_img = cv2.dilate(gray_matrix, kernel_dilate, iterations=1)
    eroded_img1 = cv2.erode(dilated_img, kernel_dilate, iterations=1)
    eroded_img = cv2.erode(eroded_img1, kernel_erode, iterations=1)
    num_labels, labels = cv2.connectedComponents(eroded_img)
    num_connected_components = num_labels - 1

    # plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(img_title + '\n pred_num=' + str(num_connected_components), fontsize=20)
    plt.imshow(gray_matrix)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(str(num_connected_components))
    plt.imshow(eroded_img)
    plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title(str(num_connected_components))
    # plt.imshow(eroded_img)
    # plt.axis('off')

    plt.show()

    return num_connected_components


def count_area_num_without_figure(bool_input, img_title, kernel_size=47):
    # 不画图没用到img_title
    gray_matrix = np.array(bool_input, dtype=np.uint8) * 255
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * 5 + 1, 2 * 5 + 1), (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
    dilated_img = cv2.dilate(gray_matrix, kernel_dilate, iterations=1)
    eroded_img1 = cv2.erode(dilated_img, kernel_dilate, iterations=1)
    eroded_img = cv2.erode(eroded_img1, kernel_erode, iterations=1)
    num_labels, labels = cv2.connectedComponents(eroded_img)
    num_connected_components = num_labels - 1

    return num_connected_components
