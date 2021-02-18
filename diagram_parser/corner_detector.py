# Python programe to illustrate
# corner detection with
# Harris Corner Detection Method

# organizing imports
import cv2.cv2 as cv2
import numpy as np
from utils.tools import freedman_diaconis_bins, otsus_threshold
import matplotlib.pyplot as plt
from diagram_parser.point_detector import remove_text
from diagram_parser.point_detector import get_connected_components, imshow_components


def distance(x1, y1, x2, y2):
    return np.linalg.norm((x2 - x1, y2 - y1))


def get_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked = remove_text(gray)
    corners = cv2.cornerHarris(masked, 2, 3, 0.04)
    dilated_corners = cv2.dilate(corners, None)
    filtered_dest = np.zeros_like(dilated_corners)
    filtered_dest[dilated_corners > 0.04 * dilated_corners.max()] = 255
    uint8_filtered_dest = np.uint8(filtered_dest)
    _, components, _, centroids = get_connected_components(uint8_filtered_dest)

    return centroids[1:]


def draw_corners(image, points):
    image_copy = image.copy()
    for point in points:
        int_point = map(lambda f: int(f), point)  # convert coordinates to ints
        cv2.circle(image_copy, tuple(int_point), 2, [0, 255, 0], thickness=-1)
    return image_copy


# img = cv2.imread('../aaai/038.png')
# corners = get_corners(img)
# image_with_corners = draw_corners(img, corners[1:])
# cv2.imshow('image with corners', image_with_corners)
# cv2.waitKey()
