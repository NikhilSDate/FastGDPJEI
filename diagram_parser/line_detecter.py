import cv2.cv2 as cv2
import numpy as np
from experiments.params import Params
from numpy import arctan2
from math import sqrt
from math import pi
import os
from diagram_parser.text_detector import remove_text
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def preprocess(img):
    # TODO: TUNE THESE PARAMETERS
    img = cv2.erode(img, kernel=np.ones((2, 2)), iterations=1)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def inclination(theta):
    if theta > np.pi / 2:
        return theta - (np.pi / 2)
    else:
        return theta + (np.pi / 2)


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

def line_length(endpoints):
    x1 = endpoints[0]
    y1 = endpoints[1]
    x2 = endpoints[2]
    y2 = endpoints[3]
    return sqrt((x2-x1)**2+(y2-y1)**2)

def clustering_filter(lines, image_size):
    line_lengths = [line_length(endpoints) for endpoints in lines]
    hesse_lines = [np.array(hesse_normal_form(endpoints)) for endpoints in lines]
    x = np.array([line[0] for line in hesse_lines])
    y = np.array([line[1] for line in hesse_lines])
    x /= max(image_size[0], image_size[1])
    y /= 2 * pi
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.075, affinity=cylindrical_similarity)
    eps = Params.params['line_detector_clustering_eps']
    clustering = DBSCAN(eps=eps, min_samples=1, metric=cylindrical_similarity)
    clustering.fit(list(zip(x, y)))
    # plt.scatter(x, y, c=clustering.labels_.astype(float))
    # plt.show()
    cluster_dict = {}
    for idx, label in enumerate(clustering.labels_):
        if label in cluster_dict:
            cluster_dict[label].append((hesse_lines[idx], line_lengths[idx]))
        else:
            cluster_dict[label] = [(hesse_lines[idx], line_lengths[idx])]
    averaged_lines = []
    for _, lines in cluster_dict.items():
        averaged_lines.append(average_lines(lines))
    return averaged_lines


def filter_lines_p(lines_p, image_size):
    accepted_line_groups = list()
    while len(lines_p) > 0:
        indices_to_remove = [0]
        line = lines_p[0]
        current_set = [line]
        for idx, line2 in enumerate(lines_p[1:], start=1):
            if close_enough_p(line, line2, image_size):
                indices_to_remove.append(idx)
                current_set.append(line2)
        accepted_line_groups.append(current_set)
        lines_p = np.delete(lines_p, indices_to_remove, axis=0)
    filtered_lines_p = []
    for line_group in accepted_line_groups:
        first_line = line_group[0]
        group_inclination = inclination(hesse_normal_form(first_line)[1])
        x1, y1, x2, y2 = line_group[0]

        if np.pi / 4 < group_inclination < 3 * np.pi / 4:
            for line in line_group:
                if line[1] < y1:
                    y1 = line[1]
                    x1 = line[0]
                if line[3] > y2:
                    y2 = line[3]
                    x2 = line[2]
        else:
            for line in line_group:
                if line[0] < x1:
                    x1 = line[0]
                    y1 = line[1]
                if line[2] > x2:
                    x2 = line[2]
                    y2 = line[3]
        filtered_lines_p.append([x1, y1, x2, y2])
    return filtered_lines_p


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
    # PARAM line_detector_close_enough_angle_threshold
    # PARAM line_detector_close_enough_rho_threshold
    angle_thresh = Params.params['line_detector_close_enough_angle_threshold']
    rho_thresh = Params.params['line_detector_close_enough_rho_threshold']
    if angle_difference < angle_thresh and rho_difference < rho_thresh * (image_size[0] + image_size[1]) / 2:
        return True
    return False
def match_close_enough(line1, line2, image_size):
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
    # TESTING_PARAM line_detector_close_enough_angle_threshold
    # TESTING_PARAM line_detector_close_enough_rho_threshold
    if angle_difference < 0.1 and rho_difference < 0.05 * (image_size[0] + image_size[1]) / 2:
        return True
    return False
def close_enough_p(line1, line2, image_size):
    hesse_line1 = hesse_normal_form(line1)
    hesse_line2 = hesse_normal_form(line2)
    return close_enough(hesse_line1, hesse_line2, image_size)
def hesse_normal_form(line):
    x1, y1, x2, y2 = line
    A = y1 - y2
    B = x2 - x1
    C = (x1-x2)*y1 + (y2-y1)*x1
    cosine = A/sqrt(A**2+B**2)
    sine = B/sqrt(A**2+B**2)
    negative_rho = C/sqrt(A**2+B**2)
    rho = -negative_rho
    theta = arctan2(sine, cosine)

    return convert_to_positive([rho, theta])
def cylindrical_similarity(l1, l2):
    rho_diff = abs(l1[0]-l2[0])
    theta_diff = abs(l1[1] - l2[1])
    theta_diff = min(theta_diff, 1-theta_diff)
    return sqrt(rho_diff**2+theta_diff**2)
def average_lines(lines_with_weights):
    lines = [line_with_weight[0] for line_with_weight in lines_with_weights]
    weights = [line_with_weight[1] for line_with_weight in lines_with_weights]

    min_theta = np.min(lines, axis=0)[1]
    max_theta = np.max(lines, axis=0)[1]
    if max_theta-min_theta > np.pi/2:
        fixed_lines = []
        for line in lines:
            if 2*pi-line[1] < line[1]:
                fixed_lines.append([line[0], line[1]-2*pi])
            else:
                fixed_lines.append([line[0], line[1]])
        return np.average(fixed_lines, axis=0)
    else:
        return np.average(lines, axis=0)


def get_hough_lines(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: TUNE THESE PARAMETERS
    canny_params = Params.params['line_detector_canny_params']
    edges = cv2.Canny(img, canny_params[0], canny_params[1], apertureSize=canny_params[2])
    lines = cv2.HoughLines(edges, 1, np.pi / 45, 40)
    return lines
def get_hough_lines_p(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # PARAM line_detector_hough_p_params
    rho, theta, thresh, minLineLength, maxLineGap = Params.params['line_detector_hough_p_params']

    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=thresh, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        lines = [line[0] for line in lines]
    else:
        lines = []
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


def get_filtered_lines(img, filter_method='cluster'):
    mode = Params.params['line_detector_mode']
    if mode == 'hough':
        hough_lines = get_hough_lines(img)
        if hough_lines is None:
            return np.array([])
        else:
            hough_lines = [line[0] for line in hough_lines]  # remove double array
            filtered_lines = filter_lines(hough_lines, img.shape)
            return filtered_lines
    elif mode == 'hough_p_hesse':
        cv2.destroyAllWindows()
        resize_image = Params.params['resize_image_if_too_big']
        resize_dim = Params.params['resize_dim']
        max_dimension = max(img.shape[0], img.shape[1])
        if max_dimension > resize_dim and resize_image:
            factor = resize_dim/max_dimension
            img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        else:
            factor = 1
        hough_lines_p = get_hough_lines_p(img)
        hough_lines_p = np.multiply(hough_lines_p, 1 / factor)
        if hough_lines_p is None or len(hough_lines_p) == 0:
            return np.array([])
        else:
            if filter_method == 'cluster':
                filtered_lines = clustering_filter(hough_lines_p, np.array(np.array(img.shape)*1/factor))
                # cv2.imshow('lines', draw_lines(cv2.resize(img, (0, 0), fx=1/factor, fy=1/factor), filtered_lines))
                # cv2.waitKey()
                return filtered_lines
            else:
                filtered_lines = filter_lines_p(hough_lines_p, np.array(np.array(img.shape)*1/factor))
                hesse_lines = [np.array(hesse_normal_form(endpoints_line)) for endpoints_line in filtered_lines]
                return hesse_lines

# diagram = cv2.imread('../experiments/data/images/0032.png')
# gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY)
# gray = remove_text(gray)
# new_lines = get_filtered_lines(gray, 'cluster')
# old_lines = get_filtered_lines(gray, 'all')
# cv2.imshow('new lines', draw_lines(diagram, new_lines))
# cv2.imshow('old lines', draw_lines(diagram.copy(), old_lines))
# cv2.waitKey()
# import os
# import time
# count = 0
# selecting = 0
# totalstart = time.time()
# for filename in os.listdir('../symbols/'):
#         if filename.endswith('.png'):
#             diagram = cv2.imread('../symbols/'+filename)
#             gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY)
#             gray = remove_text(gray)
#             old_lines = get_filtered_lines(gray)
#             new_lines = get_filtered_lines(gray, 'cluster')
#             cv2.imshow('new lines', draw_lines(diagram, new_lines))
#             cv2.imshow('old lines', draw_lines(diagram.copy(), old_lines))
#             cv2.waitKey()
#             # plt.show()
#             cv2.destroyAllWindows()
#             print(f'files done: {count}\r')
#             print(filename)

# plt.xlim(0, 1)
# plt.ylim(0, 1)
# filtered_lines = get_filtered_lines(gray)
# cv2.imshow('lines', draw_lines(diagram, averaged_lines))
# cv2.imshow('old lines', draw_lines(diagram.copy(), filtered_lines))
# # plt.scatter(x, y)
# # plt.show()
# cv2.waitKey()
