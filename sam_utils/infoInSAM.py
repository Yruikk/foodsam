import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_input_point_for_layering():
    input_point = np.array([[675, 15],
                            [675, 55],
                            [105, 129], [1220, 126],
                            [21, 168], [1259, 151]
                            ])  # 1280, 720 from 左上角
    return input_point


def get_triangular_region(layer):
    # region1: (0, 0) --> (0, a) --> (b, 0) --> (0, 0)
    # region2: (0, c) --> (0, 1280) --> (d, 1280) --> (0, c)
    # region3: (e, 1280) --> (720, 1280) --> (720, f) --> (e, 1280)
    # region4: (g, 0) --> (720, h) --> (720, 0) --> (g, 0)
    if layer == 1:
        triangular_region = [95, 719, 1150, 719, 620, 1060, 600, 260]
    elif layer == 2:
        triangular_region = [130, 400, 1050, 400, 650, 1100, 625, 200]
    elif layer == 3:
        triangular_region = [190, 320, 1050, 310, 675, 1090, 650, 115]
    elif layer == 4:
        triangular_region = [240, 290, 1100, 215, 719, 1279, 719, 1]

    return triangular_region


def set_triangular_region_false(bool_array, r_list):
    # region1: (0, 0) --> (0, a) --> (b, 0) --> (0, 0)
    # region2: (0, c) --> (0, 1280) --> (d, 1280) --> (0, c)
    # region3: (e, 1280) --> (720, 1280) --> (720, f) --> (e, 1280)
    # region4: (g, 0) --> (720, h) --> (720, 0) --> (g, 0)
    m, n = bool_array.shape
    a, b = r_list[0], r_list[1]
    c, d = r_list[2], r_list[3]
    e, f = r_list[4], r_list[5]
    g, h = r_list[6], r_list[7]
    # Set the upper left triangle to false
    for i in range(b + 1):
        for j in range(a + 1):
            if i / b + j / a <= 1:
                bool_array[i, j] = False
    # Set the upper right triangle to false
    for i in range(d + 1):
        for j in range(c, n):
            if i / d + (n - j) / (n - c) <= 1:
                bool_array[i, j] = False
    # Set the lower right triangle to false
    for i in range(e, m):
        for j in range(f, n):
            if (m - i) / (m - e) + (n - j) / (n - f) <= 1:
                bool_array[i, j] = False
    # Set the lower left triangle to false
    for i in range(g, m):
        for j in range(h + 1):
            if (m - i) / (m - g) + j / h <= 1:
                bool_array[i, j] = False

    return bool_array


def set_leftright_region_false(bool_array, r_list):
    m, n = bool_array.shape
    left, right = r_list[0], r_list[1]
    # Set the left rectangle to false
    for i in range(m):
        for j in range(left + 1):
            bool_array[i, j] = False
    # Set the right rectangle to false
    for i in range(m):
        for j in range(right, n):
            bool_array[i, j] = False

    return bool_array


def set_up_region_false(bool_array, r_list):
    m, n = bool_array.shape
    up = r_list[0]
    # Set the up rectangle to false
    for i in range(up):
        for j in range(n):
            bool_array[i, j] = False

    return bool_array


def get_seg_type(food_class, layer):
    # if food_class == 'chips' and layer == 2:
    #     point_or_box = 'box'
    # else:
    #     point_or_box = 'point'
    point_or_box = 'point'

    return point_or_box


def merge_point(pos_point, neg_point):
    if neg_point is not None:
        input_point = np.vstack([pos_point, neg_point])
        input_label = np.ones(input_point.shape[0])
        input_label[-1] = 0  # 0 for background, 1 for object
    else:
        input_point = pos_point
        input_label = np.ones(input_point.shape[0])

    return input_point, input_label


def get_input_points(image, layer, food_class):
    if food_class == 'blueberry' or food_class == 'eggtart' or food_class == 'mango' or food_class == 'petal' \
            or food_class == 'strawberry':
        if layer == 1:
            neg_point = None
            pos_point = np.array([[20, 20], [445, 25], [808, 20], [1260, 20],
                                  [129, 127], [1146, 155],
                                  [89, 453], [1195, 448],
                                  [20, 700], [445, 719], [808, 719], [1260, 700]
                                  ])  # 1280, 720 from 左上角
        elif layer == 2:
            neg_point = None
            pos_point = np.array([[20, 20], [445, 12], [808, 12], [1260, 20],
                                  [124, 127], [1116, 155],
                                  [38, 453], [1240, 448],
                                  [20, 700], [445, 716], [808, 716], [1260, 700]
                                  ])  # 1280, 720 from 左上角
        elif layer == 3:
            neg_point = None
            if food_class == 'petal':
                pos_point = np.array([[445, 43], [808, 43],
                                      [144, 127], [1135, 155],
                                      [28, 365], [1246, 334],
                                      [25, 700], [445, 710], [808, 710], [1245, 710]
                                      ])  # 1280, 720 from 左上角
            else:
                pos_point = np.array([[445, 43], [808, 43],
                                      [144, 127], [1135, 155],
                                      [28, 365], [1246, 334],
                                      [445, 710], [808, 710]
                                      ])  # 1280, 720 from 左上角
        elif layer == 4:
            neg_point = None
            pos_point = np.array([[445, 65], [808, 60],
                                  [162, 127], [1188, 155],
                                  [20, 336], [1250, 248],
                                  [25, 700], [445, 710], [808, 710], [1245, 710]
                                  ])  # 1280, 720 from 左上角

        input_point, input_label = merge_point(pos_point, neg_point)
        kernel_size = 15

    elif food_class == 'cookieTart':
        if layer == 1:
            neg_point = neg_point_by_color(image, 'yellow')
            pos_point = np.array([[445, 25], [808, 20],
                                  [129, 127], [1146, 155],
                                  [89, 453], [1195, 448],
                                  [445, 719], [808, 719],
                                  ])  # 1280, 720 from 左上角
        elif layer == 2:
            neg_point = neg_point_by_color(image, 'yellow')
            pos_point = np.array([[20, 20], [445, 12], [808, 12], [1260, 20],
                                  [124, 127], [1116, 155],
                                  [38, 453], [1240, 448],
                                  [20, 700], [445, 716], [808, 716], [1260, 700]
                                  ])  # 1280, 720 from 左上角
        elif layer == 3:
            neg_point = None
            pos_point = np.array([[445, 43], [808, 43],
                                  [144, 127], [1135, 155],
                                  [28, 365], [1246, 334],
                                  [445, 710], [808, 710]
                                  ])  # 1280, 720 from 左上角
        elif layer == 4:
            neg_point = None
            pos_point = np.array([[445, 65], [808, 60],
                                  [162, 127], [1188, 155],
                                  [20, 336], [1250, 248],
                                  [25, 700], [445, 710], [808, 710], [1245, 710]
                                  ])  # 1280, 720 from 左上角

        input_point, input_label = merge_point(pos_point, neg_point)
        kernel_size = 10

    elif food_class == 'boat':
        if layer == 1:
            neg_point = neg_point_by_color(image, 'yellow')
            pos_point = np.array([[445, 25], [808, 20],
                                  [129, 127], [1146, 155],
                                  [89, 453], [1195, 448],
                                  [445, 719], [808, 719],
                                  ])  # 1280, 720 from 左上角
        elif layer == 2:
            neg_point = None
            pos_point = np.array([[124, 127], [1116, 155],
                                  [38, 453], [1240, 448],
                                  ])  # 1280, 720 from 左上角
        elif layer == 3:
            neg_point = None
            pos_point = np.array([[445, 43], [808, 43],
                                  [144, 127], [1135, 155],
                                  [28, 365], [1246, 334],
                                  [445, 710], [808, 710]
                                  ])  # 1280, 720 from 左上角
        elif layer == 4:
            neg_point = None
            pos_point = np.array([[445, 65], [808, 60],
                                  [162, 127], [1188, 155],
                                  [20, 336], [1250, 248],
                                  [25, 700], [445, 710], [808, 710], [1245, 710]
                                  ])  # 1280, 720 from 左上角

        input_point, input_label = merge_point(pos_point, neg_point)
        kernel_size = 10

    elif food_class == 'chips':
        if layer == 1:
            neg_point = None
            pos_point = np.array([[445, 25], [808, 20],
                                  [129, 127], [1146, 155],
                                  [89, 453], [1195, 448],
                                  [445, 719], [808, 719],
                                  ])  # 1280, 720 from 左上角
        elif layer == 2:
            neg_point = None
            pos_point = np.array([[445, 12], [808, 12],
                                  [124, 127], [1116, 155],
                                  [38, 453], [1240, 448],
                                  [20, 700], [445, 716], [808, 716], [1260, 700]
                                  ])  # 1280, 720 from 左上角
        elif layer == 3:
            neg_point = None
            pos_point = np.array([[445, 43], [808, 43],
                                  [144, 127], [1135, 155],
                                  [28, 365], [1246, 334],
                                  [25, 700], [445, 710], [808, 710], [1245, 710]
                                  ])  # 1280, 720 from 左上角
        # elif layer == 4:
        #     neg_point = None
        #     pos_point = np.array([[445, 65], [808, 60],
        #                           [162, 127], [1188, 155],
        #                           [20, 336], [1250, 248],
        #                           [25, 700], [445, 710], [808, 710], [1245, 710]
        #                           ])  # 1280, 720 from 左上角

        input_point, input_label = merge_point(pos_point, neg_point)
        kernel_size = 10

    return input_point, input_label, kernel_size


def neg_point_by_color(image, color):
    if color == 'yellow':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([int(30 / 2), int(0.2 * 255), int(0.7 * 255)])
        upper_yellow = np.array([int(40 / 2), int(0.6 * 255), int(0.9 * 255)])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        erode_kernel = np.ones((15, 15), np.uint8)
        eroded_mask = cv2.erode(mask, erode_kernel, iterations=1)
        center = np.array([int(720 / 2), int(1280 / 2)])
        coords_255 = np.argwhere(eroded_mask == 255)
        if coords_255.size == 0:
            reversed_coord = None
        else:
            distances = np.linalg.norm(coords_255 - center, axis=1)
            nearest_index = np.argmin(distances)
            nearest_coord = coords_255[nearest_index]
            reversed_coord = nearest_coord[::-1]

    return reversed_coord


def box_by_color(image, food_class):
    if food_class == 'chips':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([int(10 / 2), int(0.2 * 255), int(0.5 * 255)])
        upper_yellow = np.array([int(50 / 2), int(0.6 * 255), int(0.9 * 255)])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x + w > x_max:
                x_max = x + w
            if y + h > y_max:
                y_max = y + h
        if y_min < 40:
            y_min = 40
        elif y_max > 630:
            y_max = 720

    return np.array([x_min, y_min, x_max, y_max])


def get_input_boxs(image, layer, food_class):
    # x0, y0 = box[0], box[1]
    # w, h = box[2] - box[0], box[3] - box[1]
    if food_class == 'chips':
        if layer == 2:
            input_box = box_by_color(image, food_class)
        elif layer == 3:
            input_box = box_by_color(image, food_class)

    return input_box


if __name__ == '__main__':
    m = 10
    n = 15
    true_matrix = np.ones((m, n), dtype=bool)
    r_list = [3, 4, 10, 2, 6, 10, 8, 3]
    new_mat = set_triangular_region_false(true_matrix, r_list)
    pass
