from diagram_parser.line_detecter import get_filtered_lines, draw_lines
from diagram_parser.corner_detector import get_corners, draw_corners
from diagram_parser.point_detector import get_text_component_centroids, remove_text
from utils.tools import round_up_to_multiple, round_down_to_multiple
import cv2.cv2 as cv2
import numpy as np
from math import sin, cos
import itertools
import networkx as nx


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


def get_merged_intersections(lines, image_shape, merge_threshold = 5):
    intersections = dict()
    line_pairs = itertools.combinations(lines, 2)
    indices = itertools.combinations(range(len(lines)), 2)
    for idx, pair in zip(indices, line_pairs):
        intersection_point = get_intersection(pair[0], pair[1])
        if intersection_point is not None:
            if (0 <= intersection_point[0] <= image_shape[1]) and (0 <= intersection_point[1] <= image_shape[0]):
                intersections[idx] = intersection_point
    # merge similar intersections with a weird algorithm
    grid = []
    for i in range(image_shape[0]):
        row = [-1]*image_shape[1]
        grid.append(row)
    merged_items = dict()
    connections = nx.Graph()
    for pair in intersections.items():
        connections.add_node(pair[0])
        coord = pair[1]
        x_range, y_range = (range(int(coord[0]-2), int(coord[0]+2)), range(int(coord[1]-2), int(coord[1]+2)))
        overlapping_indices = set()
        for y in y_range:
            for x in x_range:
                if not ((x < 0 or x >= image_shape[1]) or (y < 0 or y >= image_shape[0])):
                    if grid[y][x] != -1:
                        overlapping_indices.add(grid[y][x])

                    grid[y][x] = pair[0]

        for index in overlapping_indices:
            connections.add_edge(pair[0], index)
    components = nx.connected_components(connections)
    for component in components:
        values = []
        for node in component:
            values.append(intersections[node])
        merged_items[tuple(component)] = values
    final_merged_items=dict()
    for item in merged_items.items():
        intersecting_lines=set()
        for line_pair in item[0]:
            intersecting_lines.add(line_pair[0])
            intersecting_lines.add(line_pair[1])
        final_merged_items[tuple(item[1][0])] = tuple(intersecting_lines)
    return final_merged_items


def get_strong_pairs(set1, set2, comparator):
    if isinstance(set2, dict):
        entry_list = list(set2.items())
    else:
        entry_list = set2
    product = itertools.product(set1, entry_list)
    sorted_products = sorted(product, key=comparator)
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
    flattened_triples = list()
    for item in accepted_triples:
        corner_coords = item[0][0]
        int_coords = item[0][1]
        centroid = item[1]
        flattened_triples.append((corner_coords, int_coords, centroid))
    return flattened_triples


def coord_intersection_distance(corner_intersection_pair):
    corner_coords = corner_intersection_pair[0]
    intersection_coords = corner_intersection_pair[1][0]
    return np.linalg.norm(np.subtract(corner_coords,  intersection_coords))
def coord_pair_distance(coord_pair):
    corner_coords = coord_pair[0]
    centroid_coords = coord_pair[1]
    return np.linalg.norm(np.subtract(corner_coords,  centroid_coords))



def corner_intersection_text_distance(triple):
    corner_coords = triple[0][0]
    intersection_coords = triple[0][1][0]
    text_centroid = triple[1]
    corner_to_intersection = np.linalg.norm(intersection_coords - corner_coords)
    intersection_to_text = np.linalg.norm(text_centroid - intersection_coords)
    text_to_corner = np.linalg.norm(corner_coords - text_centroid)
    return corner_to_intersection + intersection_to_text + text_to_corner
def get_weak_structures(corners, intersections, centroids, strong_items):
    weak_corners_set = set([tuple(corner) for corner in corners])
    weak_ints_set = set(intersections.items())
    weak_centroids_set = set([tuple(centroid) for centroid in centroids])
    for item in strong_items:
        weak_corners_set.remove(tuple(item[0]))
        weak_ints_set.remove(item[1])
        weak_centroids_set.remove(tuple(item[2]))
    weak_corners_list = [np.array(weak_corner) for weak_corner in weak_corners_set]
    weak_centroids_list = [np.array(weak_centroid) for weak_centroid in weak_centroids_set]

    return weak_corners_list, weak_ints_set, weak_centroids_list
image = cv2.imread('../aaai/009.png')
corner_int_img = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = remove_text(gray)
lines = get_filtered_lines(filtered)
image_with_lines = draw_lines(image, lines)
# cv2.imshow('lines', image_with_lines)
# cv2.waitKey()
intersections = get_merged_intersections(lines, image.shape)
corners = get_corners(image)
image_with_corners = draw_corners(image, corners)
cv2.imshow('corners', image_with_corners)
cv2.waitKey()
text_centroids = get_text_component_centroids(image)
strong_pairs = get_strong_pairs(corners, intersections, comparator=coord_intersection_distance)
strong_triples = get_strong_triples(strong_pairs, text_centroids)
weak_corners, weak_intersections, weak_centroids = get_weak_structures(corners, intersections, text_centroids, strong_triples)
corner_int = get_strong_pairs(weak_corners, weak_intersections, comparator=coord_intersection_distance)
corner_cent = get_strong_pairs(weak_corners, weak_centroids, comparator=coord_pair_distance)
cent_int = get_strong_pairs(weak_centroids, weak_intersections, comparator = coord_intersection_distance)
print(corner_int, corner_cent, cent_int)
for strong_triple in strong_triples:
    cornerX, cornerY = int(strong_triple[0][0]), int(strong_triple[0][1])
    intX, intY = int(strong_triple[1][0][0]), int(strong_triple[1][0][1])
    textX, textY = int(strong_triple[2][0]), int(strong_triple[2][1])
    cv2.circle(image, (cornerX, cornerY), 2, [255, 0, 0], -1)
    cv2.circle(image, (intX, intY), 2, [0, 255, 0], -1)
    cv2.circle(image, (textX, textY), 2, [0, 0, 255], -1)
for pair in corner_cent:
    cornerX, cornerY = int(pair[0][0]), int(pair[0][1])
    intX, intY = int(pair[1][0]), int(pair[1][1])
    cv2.circle(corner_int_img, (cornerX, cornerY), 2, [255, 0, 0], -1)
    cv2.circle(corner_int_img, (intX, intY), 2, [0, 255, 0], -1)


cv2.imshow('triples', image)
cv2.waitKey()
cv2.imshow('corners and intersections', corner_int_img)
cv2.waitKey()
# image_with_intersections = image.copy()
# image_with_corners = draw_corners(image, corners)
# for line_indices, point in intersections.items():
#     cv2.circle(image_with_intersections, (int(point[0]), int(point[1])), 2, [0, 0, 255], -1)
#
