from diagram_parser.line_detecter import get_filtered_lines, draw_lines
from diagram_parser.corner_detector import get_corners, draw_corners
from diagram_parser.point_detector import get_text_component_centroids, remove_text
from utils.tools import round_up_to_multiple, round_down_to_multiple
import cv2.cv2 as cv2
import numpy as np
from math import sin, cos
import itertools


def is_pair_consistent(pairs, pair):
    consistent = True
    for pair2 in pairs:
        if (pair[0] == pair2[0]).all() or (pair[1][0] == pair2[1][0]):
            consistent = False
            break
    return consistent


# Checks for overlap
def is_triple_consistent(triples, triple):
    consistent = True
    for triple2 in triples:
        if ((triple2[0][0] == triple[0][0]).all() or (triple2[0][1] == triple[0][1])) or (
                triple2[1] == triple[1]).all():
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


def get_all_intersections(lines, image_shape, merge_threshold = 5):
    intersections = dict()
    line_pairs = itertools.combinations(lines, 2)
    indices = itertools.combinations(range(len(lines)), 2)
    for idx, pair in zip(indices, line_pairs):
        intersection_point = get_intersection(pair[0], pair[1])
        if intersection_point is not None:
            if (0 <= intersection_point[0] <= image_shape[1]) and (0 <= intersection_point[1] <= image_shape[0]):
                intersections[idx] = intersection_point
    # merge similar intersections with a weird algorithm
    merged_intersections = dict()
    for intersection in intersections.items():
        y = intersection[1][0]
        x = intersection[1][1]
        x_up = round_up_to_multiple(x, merge_threshold)
        x_down = round_down_to_multiple(x, merge_threshold)
        y_up = round_up_to_multiple(y, merge_threshold)
        y_down = round_down_to_multiple(y, merge_threshold)
        print(x_up, x_down, y_up, y_down)
        combinations = ((y_up, x_up), (y_up, x_down), (y_down, x_up), (y_down, x_down))
        merged = False
        for combination in combinations:
            if combination in merged_intersections:
                merged_intersections[combination] = set(intersection[0]).union(set(merged_intersections[combination]))
                merged = True
        if not merged:
            for combination in combinations:
                print(combination)
                merged_intersections[combination] = intersection[0]
    merged_items = list(merged_intersections.items())
    final_merged_items = dict()
    # Iterate over batches of four items
    for i in range(0, len(merged_items), 4):
        item1, item2, item3, item4 = merged_items[i], merged_items[i+1], merged_items[i+2], merged_items[i+3]
        union = set().union(item1[1], item2[1], item3[1], item4[1])
        final_merged_items[item1[0]] = union
    print(final_merged_items)
    return final_merged_items


def get_strong_pairs(corners, intersections):
    entry_list = list(intersections.items())
    product = itertools.product(corners, entry_list)
    sorted_products = sorted(product, key=corner_intersection_distance)
    accepted_pairs = []
    while sorted_products:
        pair = sorted_products.pop(0)
        if is_pair_consistent(accepted_pairs, pair):
            accepted_pairs.append(pair)
    return accepted_pairs


def get_strong_triples(strong_pairs, text_coords):
    triples = list(itertools.product(strong_pairs, text_coords))
    sorted_triples = sorted(triples, key=corner_intersection_text_distance)
    accepted_triples = []
    while sorted_triples:
        triple = sorted_triples.pop(0)
        if is_triple_consistent(accepted_triples, triple):
            accepted_triples.append(triple)

    return accepted_triples


def corner_intersection_distance(corner_intersection_pair):
    corner_coords = corner_intersection_pair[0]
    intersection_coords = corner_intersection_pair[1][0]
    return np.linalg.norm(corner_coords - intersection_coords)


def corner_intersection_text_distance(triple):
    corner_coords = triple[0][0]
    intersection_coords = triple[0][1][0]
    text_centroid = triple[1]
    corner_to_intersection = np.linalg.norm(intersection_coords - corner_coords)
    intersection_to_text = np.linalg.norm(text_centroid - intersection_coords)
    text_to_corner = np.linalg.norm(corner_coords - text_centroid)
    return corner_to_intersection + intersection_to_text + text_to_corner


image = cv2.imread('../aaai/058.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = remove_text(gray)
lines = get_filtered_lines(filtered)
image_with_lines = draw_lines(image, lines)
cv2.imshow('lines', image_with_lines)
cv2.waitKey()
intersections = get_all_intersections(lines, image.shape)
corners = get_corners(image)
image_with_corners = draw_corners(image, corners)
# cv2.imshow('corners', image_with_corners)
# cv2.waitKey()
text_centroids = get_text_component_centroids(image)
strong_pairs = get_strong_pairs(corners, intersections)
strong_triples = get_strong_triples(strong_pairs, text_centroids)
for strong_triple in strong_triples:
    cornerX, cornerY = int(strong_triple[0][0][0]), int(strong_triple[0][0][1])
    intX, intY = int(strong_triple[0][1][0][0]), int(strong_triple[0][1][0][1])
    textX, textY = int(strong_triple[1][0]), int(strong_triple[1][1])
    cv2.circle(image, (cornerX, cornerY), 2, [255, 0, 0], -1)
    cv2.circle(image, (intX, intY), 2, [0, 255, 0], -1)
    cv2.circle(image, (textX, textY), 2, [0, 0, 255], -1)
cv2.imshow('triples', image)
cv2.waitKey()
# image_with_intersections = image.copy()
# image_with_corners = draw_corners(image, corners)
# for line_indices, point in intersections.items():
#     cv2.circle(image_with_intersections, (int(point[0]), int(point[1])), 2, [0, 0, 255], -1)
#
