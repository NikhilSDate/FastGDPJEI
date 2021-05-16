import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin


def preprocess(img):
    # TODO: TUNE THESE PARAMETERS
    img = cv2.erode(img, kernel=np.ones((2, 2)), iterations=1)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def inclination(alpha):
    if alpha > np.pi / 2:
        return alpha - (np.pi / 2)
    else:
        return alpha + (np.pi / 2)


# def filter_lines(lines, index):
#     indices_to_remove = list()
#     line = lines[index]
#     angle_1 = inclination(alpha=line[0][1])
#     for i in range(index + 1, lines.shape[0]):
#         angle_2 = inclination(alpha=lines[i][0][1])
#         angle_between_lines = abs(angle_2 - angle_1)
#         if angle_between_lines > np.pi / 2:
#             angle_between_lines = np.pi - angle_between_lines
#         # remove lines for which the angle is less than ~5 degrees and the difference between the rhos is less than
#         # 20 pixels
#         if angle_between_lines < 0.1 and abs(abs(lines[i][0][0]) - abs(line[0][0])) < 20:
#             indices_to_remove.append(i)
#     filtered_lines = np.delete(lines, indices_to_remove, axis=0)
#
#     if index + 1 == len(filtered_lines):
#         return filtered_lines
#     else:
#         return filter_lines(filtered_lines, index + 1)


def filter_lines(lines, image_size):
    accepted_line_groups = list()
    while len(lines) > 0:
        indices_to_remove = [0]
        line = lines[0]
        current_set = [line]
        for idx, line2 in enumerate(lines[1:], start=1):
            if close_enough(line, line2, image_size):
                indices_to_remove.append(idx)
                current_set.append(line2)
        accepted_line_groups.append(current_set)
        lines = np.delete(lines, indices_to_remove, axis=0)
    filtered_lines = []
    for line_group in accepted_line_groups:
        if (np.array(line_group) < 0).any():
            filtered_lines.append(line_group[0])
        else:
            average_line = np.average(line_group, axis=0, weights=[1 / x for x in range(1, len(line_group) + 1)])
            filtered_lines.append(average_line)

    return filtered_lines


def convert_to_positive(line):
    rho = line[0]
    theta = line[1]
    if rho < 0:
        rho = -rho
        theta = np.pi + theta
    return rho, theta


def close_enough(line1, line2, image_size):
    rho1, theta1 = convert_to_positive(line1)
    rho2, theta2 = convert_to_positive(line2)
    angle_difference = abs(theta1 - theta2)
    if angle_difference < np.pi / 2:
        pass
    elif np.pi / 2 < angle_difference <= np.pi:
        angle_difference = np.pi - angle_difference
    elif np.pi <= angle_difference <= 3 * np.pi / 2:
        angle_difference = angle_difference - np.pi
    elif 3 * np.pi / 2 <= angle_difference < 2 * np.pi:
        angle_difference = 2 * np.pi - angle_difference
    rho_difference = abs(rho1 - rho2)
    # TODO: TUNE THESE PARAMETERS
    if angle_difference < 0.1 and rho_difference < 0.075*(image_size[0]+image_size[1])/2:
        return True
    return False


def get_hough_lines(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: TUNE THESE PARAMETERS
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 45, 40)
    return lines


def draw_lines(img, lines):
    img_copy = img.copy()
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_copy

    # plt.hist(distances_list,density=True,bins=30)
    # plt.show()


def get_filtered_lines(img):
    hough_lines = get_hough_lines(img)
    if hough_lines is None:
        return np.array([])
    else:
        hough_lines = [line[0] for line in hough_lines]  # remove double array
        filtered_lines = filter_lines(hough_lines, img.shape)
        return filtered_lines


# image = cv2.imread('../aaai/022.png')
#
# lines = get_hough_lines(image)
#
# cleaned_lines = [line[0] for line in lines]
# print(cleaned_lines)
# filtered_lines = get_filtered_lines(image)
# cv2.imshow('raw lines', draw_lines(image, cleaned_lines))
# cv2.imshow('old_algorithm', draw_lines(image, filtered_lines))
# x, y = zip(*cleaned_lines)
# plt.scatter(x, y, )
# plt.xlim(-200, 200)
# plt.ylim(-np.pi, np.pi)
# # plt.show()
# cv2.waitKey()
