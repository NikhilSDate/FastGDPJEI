import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


class LineDetector:

    def __init__(self, fp):
        self.img = cv2.imread(fp)

    def preprocess(self):
        blur=cv2.bilateralFilter(self.img, 9, 75, 75)
        self.img=blur

    def inclination(self, alpha):
        if alpha > np.pi / 2:
            return alpha - (np.pi / 2)
        else:
            return alpha + (np.pi / 2)

    def filter_lines(self, lines, index):
        indices_to_remove = list()
        line = lines[index]
        angle_1 = self.inclination(alpha=line[0][1])
        for i in range(index + 1, lines.shape[0]):
            angle_2 = self.inclination(alpha=lines[i][0][1])
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
            return self.filter_lines(filtered_lines, index + 1)

    def get_hough_lines(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/45 , 40)
        return lines

    def draw_lines(self, lines):
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return self.img

    # plt.hist(distances_list,density=True,bins=30)
    # plt.show()
    def get_filtered_lines(self):
        hough_lines = line_detector.get_hough_lines()
        filtered_lines = line_detector.filter_lines(hough_lines, 0)
        return filtered_lines


line_detector = LineDetector('../aaai/ncert2.png')
line_detector.preprocess()
filtered_lines = line_detector.get_filtered_lines()
img = line_detector.draw_lines(filtered_lines)

cv2.imshow('image', img)
cv2.waitKey()
