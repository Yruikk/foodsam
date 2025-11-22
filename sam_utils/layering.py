import numpy as np
import cv2


def find_layer(grayscale_matrix, start_col=340, end_col=1010, kernel_size=30):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening_img = cv2.morphologyEx(grayscale_matrix, cv2.MORPH_OPEN, kernel)

    found_row = 0
    for row_idx in range(opening_img.shape[0] - 1, -1, -1):
        if np.all(opening_img[row_idx, start_col:end_col] == 0):
            found_row = row_idx
            break

    if found_row >= 260:
        layer = 1
    elif 180 <= found_row < 260:
        layer = 2
    elif 100 <= found_row < 180:
        layer = 3
    else:
        layer = 4

    return opening_img, layer