from diagram_parser.line_detecter import get_filtered_lines, draw_lines
from diagram_parser.corner_detector import get_corners, draw_corners
from diagram_parser.text_detector import text_components_with_centroids, remove_text
import cv2.cv2 as cv2
import numpy as np
from math import sin, cos, sqrt
import itertools
import networkx as nx
from collections import OrderedDict
from nn.character_predictor import CharacterPredictor
from diagram_parser.searchnode import SearchNode
from diagram_parser.primitive import Primitive
from diagram_parser.primitivegroup import PrimitiveGroup
from diagram_parser.diagram_interpretation import Interpretation
from diagram_parser.point import Point
from queue import Queue
from sklearn.cluster import AgglomerativeClustering, DBSCAN

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
    # corner coordinates are numpy arrays so the .all() others are tuples
    for triple2 in triples:
        if ((triple2[0][0] == triple[0][0]).all() or (triple2[0][1] == triple[0][1])) or (
                triple2[1] == triple[1]):
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
        # Exception is thrown if lines are parallel
        return None
def point_to_line_distance(point_coordinates, line):
    # line is in Hesse normal form
    # first converts line to Ax + By + C = 0
    rho = line[0]
    theta = line[1]
    A = cos(theta)
    B = sin(theta)
    C = -rho
    x = point_coordinates[0]
    y = point_coordinates[1]
    distance = abs(A*x + B*y + C)/sqrt(A**2+B**2)
    return distance



def get_merged_intersections(lines, image_shape):
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
        row = [-1] * image_shape[1]
        grid.append(row)
    merged_items = dict()
    connections = nx.Graph()
    for pair in intersections.items():
        connections.add_node(pair[0])
        coord = pair[1]
        x_range, y_range = (range(int(coord[0] - 2), int(coord[0] + 2)), range(int(coord[1] - 2), int(coord[1] + 2)))
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
    final_merged_items = dict()
    for item in merged_items.items():
        intersecting_lines = set()
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
    return np.linalg.norm(np.subtract(corner_coords, intersection_coords))


def coord_pair_distance(coord_pair):
    corner_coords = coord_pair[0]
    centroid_coords = coord_pair[1]
    return np.linalg.norm(np.subtract(corner_coords, centroid_coords))


def corner_intersection_text_distance(triple):
    corner_coords = triple[0][0]
    intersection_coords = triple[0][1][0]
    text_centroid = triple[1]
    corner_to_intersection = np.linalg.norm(np.subtract(intersection_coords, corner_coords))
    intersection_to_text = np.linalg.norm(np.subtract(text_centroid, intersection_coords))
    text_to_corner = np.linalg.norm(np.subtract(corner_coords, text_centroid))
    return corner_to_intersection + intersection_to_text + text_to_corner


def get_weak_primitives(corners, intersections, centroids, strong_items):
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


def get_primitives(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = remove_text(gray)

    lines = get_filtered_lines(filtered)
    image_with_lines = draw_lines(image, lines)
    intersections = get_merged_intersections(lines, image.shape)
    corners = get_corners(image)
    image_with_corners = draw_corners(image, corners)
    text_regions = text_components_with_centroids(image)
    return corners, lines, intersections, text_regions


image = cv2.imread('../aaai/015.png')
corner_int_img = image.copy()

corners, lines, intersections, text_regions = get_primitives(image)
primitives = list()
print(lines)
for idx, corner in enumerate(corners):
    primitives.append(Primitive((corner[0], corner[1]), 'c', idx))
for idx, intersection in enumerate(intersections.keys()):
    primitives.append(Primitive(intersection, 'i', idx))
for idx, coords in enumerate(text_regions.keys()):
    primitives.append(Primitive(coords, 't', idx))

def average_distance(point, points):
    distance = 0
    for point2 in points:
        if np.linalg.norm(np.subtract(point, point2))<40:
            distance = distance +1
    return distance
def search_for_solution(primitives, corners, lines, intersections, text_regions):
    primitive_list = [primitive.coords for primitive in primitives]
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2 * image.shape[0]).fit(primitive_list)
    clustering = DBSCAN(eps=0.1*image.shape[0], min_samples=1).fit(primitive_list)
    cluster_list = []
    num_clusters = max(clustering.labels_) + 1
    for _ in range(num_clusters):
        cluster_list.append(PrimitiveGroup())
    for idx, label in enumerate(clustering.labels_):
        cluster_list[label].add(primitives[idx])
    diagram_interpretation = Interpretation()
    for cluster in cluster_list:
        if cluster.contains('t') and cluster.contains('i'):
            point = Point()

            for text_primitive in cluster.primitives['t']:
                text_region = list(text_regions.values())[text_primitive.index]
                character = CharacterPredictor().predict_character(text_region)
                point.add_label(character)
            for intersection in cluster.primitives['i']:
                structures = list(intersections.values())[intersection.index]
                for structure in structures:
                    point.add_property('lieson', structure)
            diagram_interpretation.add_point(point)
        elif cluster.contains('i'):
            point = Point()

            label = f'p{diagram_interpretation.num_points()}'
            point.add_label(label)
            for intersection in cluster.primitives['i']:
                structures = list(intersections.values())[intersection.index]
                for structure in structures:
                    point.add_property('lieson', structure)
            diagram_interpretation.add_point(point)

        elif cluster.contains('t') and cluster.contains('c'):
            line_set = set()
            corner_centroid = cluster.centroid(type='c')
            for idx, line in enumerate(lines):
                # TODO: THIS THRESHOLD IS ARBITRARY. CHOOSE A RELATIVE THRESHOLD OR REPLACE THIS WHOLE BY A DECISION TREE
                if point_to_line_distance(corner_centroid, line) < 5:
                    line_set.add(idx)
            if len(line_set) > 0:
                point = Point()
                for text_primitive in cluster.primitives['t']:
                    text_region = list(text_regions.values())[text_primitive.index]
                    character = CharacterPredictor().predict_character(text_region)
                    point.add_label(character)
                point.add_label()
                for line in line_set:
                    point.add_property('lieson', line)
                diagram_interpretation.add_point(point)


    print(diagram_interpretation)
    ordered_points = sorted(primitives, key=lambda primitive: average_distance(primitive.coords, primitive_list))
    for idx, cluster in enumerate(primitive_list):
        hue = 179 * clustering.labels_[idx] / num_clusters
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        cv2.circle(image, (int(cluster[0]), int(cluster[1])), 2, rgb.tolist(), thickness=-1)
    cv2.imshow('clustering', image)
    cv2.moveWindow('clustering', 50, 50)
    cv2.waitKey()
    # print(len(ordered_points))
    # search_queue = Queue()
    # search_queue.put(SearchNode(primitives, 0))
    # best_child = None
    # while search_queue and ordered_points:
    #     node = search_queue.get()
    #     if node.level >= len(ordered_points):
    #         break
    #     children = node.generate_children(ordered_points[node.level])
    #     sorted_children = sorted(children, key=lambda x: x.fitness())
    #
    #     # if node.level<5:
    #     #     if len(sorted_children)<2:
    #     #         for child in sorted_children:
    #     #             search_queue.put(child)
    #     #     else:
    #     #         for child in sorted_children[-2:]:
    #     #             search_queue.put(child)
    #     # else:
    #     search_queue.put(sorted_children[-1])
    #     best_child = sorted_children[-1]
    # for index, point in enumerate(best_child.points):
    #     hue = 179 * index/len(best_child.points)
    #     hsv = np.uint8([[[hue, 255, 255]]])
    #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    #     for coord in point.coord_list():
    #         cv2.circle(image, (int(coord[0]), int(coord[1])), 2, rgb.tolist(), -1)
    # cv2.imshow('test', image)
    # cv2.waitKey()
search_for_solution(primitives, corners, lines, intersections, text_regions)






















# text_centroids = text_regions.keys()
# strong_pairs = get_strong_pairs(corners, intersections, comparator=coord_intersection_distance)
# strong_triples = get_strong_triples(strong_pairs, text_centroids)
# weak_corners, weak_intersections, weak_centroids = get_weak_primitives(corners, intersections, text_centroids, [])
# corner_int = get_strong_pairs(weak_corners, weak_intersections, comparator=coord_intersection_distance)
# corner_cent = get_strong_pairs(weak_corners, weak_centroids, comparator=coord_pair_distance)
# cent_int = get_strong_pairs(weak_centroids, weak_intersections, comparator=coord_intersection_distance)
# # character = CharacterPredictor().predict_character(text_regions[strong_triples[2][2]])
# # print(strong_triples)
# # cv2.imshow('character', text_regions[strong_triples[2][2]])
# # cv2.waitKey()
# for pair in corner_int:
#     cornerX, cornerY = int(pair[0][0]), int(pair[0][1])
#     intX, intY = int(pair[1][0][0]), int(pair[1][0][1])
#     cv2.circle(corner_int_img, (cornerX, cornerY), 2, [255, 0, 0], -1)
#     cv2.circle(corner_int_img, (intX, intY), 2, [0, 255, 0], -1)
#
# image_with_intersections = image.copy()
# image_with_corners = draw_corners(image, corners)
# for point, line_indices in intersections.items():
#     cv2.circle(image_with_intersections, (int(point[0]), int(point[1])), 2, [0, 0, 255], -1)
