import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def inclination(alpha):
    if alpha > np.pi / 2:
        return alpha - (np.pi / 2)
    else:
        return alpha + (np.pi / 2)


def filter_lines(lines, index):
    indices_to_remove = list()
    line = lines[index]
    angle_1 = inclination(alpha=line[0][1])
    for i in range(index + 1, lines.shape[0]):
        angle_2 = inclination(alpha=lines[i][0][1])
        angle_between_lines = abs(angle_2 - angle_1)
        if angle_between_lines > np.pi / 2:
            angle_between_lines = np.pi - angle_between_lines

        if angle_between_lines < 0.1 and abs(abs(lines[i][0][0]) - abs(line[0][0])) < 20:
            indices_to_remove.append(i)
    filtered_lines = np.delete(lines, indices_to_remove, axis=0)
    np.set_printoptions(suppress=True)

    if index + 1 == len(filtered_lines):
        return filtered_lines
    else:
        return filter_lines(filtered_lines, index + 1)


def get_hough_lines(img):
    # TODO: fix mess of gray and colour images
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 45, 40)
    return lines


def draw_lines(img, lines):
    img_copy = img.copy()
    for line in lines:
        print('line', line)
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
    filtered_lines = filter_lines(hough_lines, 0)
    filtered_lines = [filtered_line[0] for filtered_line in filtered_lines]  # remove double array
    return filtered_lines
