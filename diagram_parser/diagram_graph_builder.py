from diagram_parser.line_detecter import get_filtered_lines, draw_lines
from diagram_parser.corner_detector import get_corners, draw_corners
from diagram_parser.point_detector import get_text_component_centroids
import cv2.cv2 as cv2
import numpy as np
from math import sin, cos
import itertools

def is_consistent(pairs, pair):
    consistent = True
    for pair2 in pairs:
        print(pair[0], pair2[0])
        if (pair[0] == pair2[0]).all() or (pair[1][1] == pair2[1][1]).all():
            consistent = False
            break
    return consistent

def get_intersection(line1, line2):
    rho1 = line1[0]
    theta1 = line1[1]
    rho2 = line2[0]
    theta2 = line2[1]
    coefficients = np.array([[cos(theta1), sin(theta1)], [cos(theta2), sin(theta2)]])
    constants = np.array([rho1, rho2])
    try:
        x = np.linalg.solve(coefficients, constants)
        return x
    except np.linalg.LinAlgError:
        # Exception is thrown if lines are parallel (no intersection point)
        return None



def get_all_intersections(lines, image_shape):
    intersections = dict()
    line_pairs = itertools.combinations(lines, 2)
    indices = itertools.combinations(range(len(lines)), 2)
    for idx, pair in zip(indices, line_pairs):
        intersection_point = get_intersection(pair[0], pair[1])
        if intersection_point is not None:
            if (0 <= intersection_point[0] <= image_shape[1]) and (0 <= intersection_point[1] <= image_shape[0]):
                intersections[idx] = intersection_point
    return intersections
def get_strong_pairs(corners, intersections):
    entry_list = list(intersections.items())
    product = itertools.product(corners, entry_list)
    sorted_products = sorted(product, key=distance_comparator)
    accepted_pairs = []
    while sorted_products:
        pair = sorted_products.pop(0)
        if is_consistent(accepted_pairs, pair):
            accepted_pairs.append(pair)
    return accepted_pairs



def distance_comparator(corner_intersection_pair):
    corner_coords = corner_intersection_pair[0]
    intersection_coords = corner_intersection_pair[1][1]
    return np.linalg.norm(corner_coords - intersection_coords)

image = cv2.imread('../aaai/ncert2.png')
image_with_intersections = image.copy()
lines = get_filtered_lines(image)
intersections = get_all_intersections(lines, image.shape)
corners = get_corners(image)
text_centroids=get_text_component_centroids(image)
print('text centroids', text_centroids)
image_with_corners = draw_corners(image, corners)
for centroid in text_centroids:
    cv2.circle(image, (int(centroid[0]), int(centroid[1])), 2, [0, 0, 255], -1)
cv2.imshow('image', image)
cv2.waitKey()
for line_indices, point in intersections.items():
    cv2.circle(image_with_intersections, (int(point[0]), int(point[1])), 2, [0, 0, 255], -1)
print('strong pairs', get_strong_pairs(corners, intersections))
image_with_lines = draw_lines(image, lines)
cv2.imshow('image', image_with_intersections)
cv2.waitKey()

cv2.imshow('image with corners', image_with_corners)
cv2.waitKey()
