from diagram_parser.line_detecter import get_filtered_lines, draw_lines, inclination
from diagram_parser.corner_detector import get_corners, draw_corners
from diagram_parser.text_detector import text_components_with_centroids, remove_text
from diagram_parser.circle_detector import detect_circles, draw_circles, remove_circles
import cv2.cv2 as cv2
import numpy as np
from math import sin, cos, sqrt
import itertools
from nn.character_predictor import CharacterPredictor
from diagram_parser.searchnode import SearchNode
from diagram_parser.primitive import Primitive
from diagram_parser.primitivegroup import PrimitiveGroup
from diagram_parser.diagram_interpretation import Interpretation
from diagram_parser.point import Point
from queue import Queue
from sklearn.cluster import DBSCAN
import sympy as sp
import skspatial.objects as skobj
from experiments.params import Params
import networkx as nx
import matplotlib.pyplot as plt


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


def calculate_intersection(structure1, structure2, corner_response_map):
    if ((structure1.shape[0] == 3) and (structure2.shape[0] == 2)) or (
            (structure1.shape[0] == 2) and (structure2.shape[0] == 3)):
        if structure1.shape[0] == 3:
            circle = structure1
            line = structure2
        else:
            circle = structure2
            line = structure1
        rho = line[0]
        theta = line[1]
        x0 = circle[0]
        y0 = circle[1]
        r = circle[2]

        A = cos(theta)
        B = sin(theta)
        C = -rho

        x = sp.Symbol('x')
        y = sp.Symbol('y')
        circle_eq = sp.Eq((x - x0) ** 2 + (y - y0) ** 2, r ** 2)

        perp_distance = point_to_line_distance((x0, y0), line)
        # PARAM circle_tangent_eps
        eps = Params.params['circle_tangent_eps']
        if abs((perp_distance - r) / r) < eps:
            perp_line_A = B
            perp_line_B = -A
            perp_line_C = A * y0 - B * x0
            perp_line_eq = sp.Eq(perp_line_A * x + perp_line_B * y + perp_line_C, 0)
            solutions = sp.solve([circle_eq, perp_line_eq], (x, y))
            if point_to_line_distance(solutions[0], line) < point_to_line_distance(solutions[1], line):
                return float(solutions[0][0].evalf()), float(solutions[0][1].evalf())
            else:
                return float(solutions[1][0].evalf()), float(solutions[1][1].evalf())
        elif perp_distance > r:
            return None
        else:
            if B == 0:
                p1 = np.array([-C / A, 0])
            else:
                p1 = np.array([0, -C / B])
            v = np.array([B, -A])
            solution1, solution2 = line_circle_intersection(p1, v, (x0, y0), r)
            valid_sols = []
            try:
                if corner_response_map[int(solution1[1])][int(solution1[0])] > 0:
                    valid_sols.append(solution1)
            except IndexError:
                pass
            try:
                if corner_response_map[int(solution2[1])][int(solution2[0])] > 0:
                    valid_sols.append(solution2)
            except IndexError:
                pass
            if len(valid_sols) == 2:
                return valid_sols
            elif len(valid_sols) == 1:
                return valid_sols[0]
            else:
                return None


    elif (structure1.shape[0] == 3) and (structure2.shape[0] == 3):
        x0 = structure1[0]
        y0 = structure1[1]
        r0 = structure1[2]

        x1 = structure2[0]
        y1 = structure2[1]
        r1 = structure2[2]
        center_dist = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if center_dist > (r0 + r1):
            return None
        solution1, solution2 = circle_circle_intersection(x0, y0, r0, x1, y1, r1)

        valid_sols = []
        try:
            if corner_response_map[int(solution1[1])][int(solution1[0])] >= 0:
                valid_sols.append(solution1)
        except IndexError:
            pass
        try:
            if corner_response_map[int(solution2[1])][int(solution2[0])] >= 0:
                valid_sols.append(solution2)
        except IndexError:
            pass

        if len(valid_sols) == 2:
            eps = Params.params['circle_tangent_eps']
            if abs(center_dist - (r0 + r1)) / (r0 + r1) < eps:
                np.mean([solution1, solution2], axis=0)

            return valid_sols
        elif len(valid_sols) == 1:
            return valid_sols[0]
        else:
            return None


    else:
        rho1 = structure1[0]
        theta1 = structure1[1]
        rho2 = structure2[0]
        theta2 = structure2[1]
        coefficients = np.array([[cos(theta1), sin(theta1)], [cos(theta2), sin(theta2)]])
        constants = np.array([rho1, rho2])
        try:
            solution = np.linalg.solve(coefficients, constants)

            if corner_response_map[int(solution[1])][int(solution[0])] > 0:
                return solution
            else:
                return None
        except IndexError:
            return None
        except np.linalg.LinAlgError:
            # Exception is thrown if lines are parallel
            return None


def line_circle_intersection(p1, v, q, r):
    a = np.dot(v, v)
    b = 2 * np.dot(v, p1 - q)
    c = np.dot(p1, p1) + np.dot(q, q) - 2 * np.dot(p1, q) - r ** 2
    coeff = [a, b, c]
    roots = np.roots(coeff)
    sol1 = p1 + roots[0] * v
    sol2 = p1 + roots[1] * v
    return tuple(sol1), tuple(sol2)


def circle_circle_intersection(x0, y0, r0, x1, y1, r1):
    # http: // paulbourke.net / geometry / circlesphere /
    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    sol1 = ((x2 + h * (y1 - y0) / d), (y2 - h * (x1 - x0) / d))
    sol2 = ((x2 - h * (y1 - y0) / d), (y2 + h * (x1 - x0) / d))
    return sol1, sol2


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
    distance = abs(A * x + B * y + C) / sqrt(A ** 2 + B ** 2)
    return distance


def point_to_circle_distance(point_coords, circle):
    center = [circle[0], circle[1]]

    r = circle[2]

    distance = abs(r - (np.linalg.norm(np.subtract(point_coords, center))))
    return distance


def get_merged_intersections(lines, circles, image_shape, corner_response_map):
    intersections = dict()
    final_intersections = dict()
    if len(lines) > 0:
        structures = lines + list(circles)
    else:
        structures = list(circles)
    structure_pairs = itertools.combinations(structures, 2)
    structure_ids = []
    for i in range(len(lines)):
        structure_ids.append(f'l{i}')
    for i in range(len(circles)):
        structure_ids.append(f'c{i}')
    indices = itertools.combinations(structure_ids, 2)
    for idx, pair in zip(indices, structure_pairs):
        intersection_point = calculate_intersection(pair[0], pair[1], corner_response_map)
        if intersection_point is not None:
            if np.ndim(intersection_point) == 1:
                if (0 <= intersection_point[0] <= image_shape[1]) and (0 <= intersection_point[1] <= image_shape[0]):
                    intersections[idx] = intersection_point
                    final_intersections[tuple(intersection_point)] = idx
            if np.ndim(intersection_point) == 2:
                valid_solutions = []
                solution1 = intersection_point[0]
                solution2 = intersection_point[1]
                if (0 <= solution1[0] <= image_shape[1]) and (
                        0 <= solution1[1] <= image_shape[0]):
                    valid_solutions.append(solution1)
                if (0 <= solution2[0] <= image_shape[1]) and (
                        0 <= solution2[1] <= image_shape[0]):
                    valid_solutions.append(solution2)
                if len(valid_solutions) > 0:
                    intersections[idx] = valid_solutions
                for valid_solution in valid_solutions:
                    final_intersections[valid_solution] = idx
    for i, circle in enumerate(circles):
        final_intersections[(circle[0], circle[1])] = (f'c{i}_center',)
    return final_intersections


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


def get_primitives_and_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = remove_text(gray)
    circles = detect_circles(filtered)
    masked = remove_circles(filtered, circles)

    lines = get_filtered_lines(masked)
    response_map, corners = get_corners(filtered)
    intersections = get_merged_intersections(lines, circles, image.shape, response_map)
    text_regions = text_components_with_centroids(image)
    return corners, lines, circles, intersections, text_regions


def get_primitives(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = remove_text(gray)
    circles = detect_circles(filtered)
    masked = remove_circles(filtered, circles)

    lines = get_filtered_lines(masked)
    return lines, circles


def parse_diagram(diagram_image, detect_labels=False):
    corners, lines, circles, intersections, text_regions = get_primitives_and_points(diagram_image)
    primitives = list()
    for idx, intersection in enumerate(intersections.keys()):
        primitives.append(Primitive(intersection, 'i', idx))
    for idx, corner in enumerate(corners):
        primitives.append(Primitive((corner[0], corner[1]), 'c', idx))
    line_dict = dict()
    circle_dict = dict()
    for idx, line in enumerate(lines):
        line_dict[f'l{idx}'] = line
    for idx, circle in enumerate(circles):
        circle_dict[f'c{idx}'] = circle
    interpretation = build_interpretation(primitives, lines, circles, intersections, text_regions, diagram_image.shape,
                                          detect_labels=detect_labels)
    interpretation.set_lines(line_dict)
    interpretation.set_circles(circle_dict)
    return interpretation, line_dict, circle_dict


# for idx, coords in enumerate(text_regions.keys()):
#     primitives.append(Primitive(coords, 't', idx))


def average_distance(point, points):
    distance = 0
    for point2 in points:
        if np.linalg.norm(np.subtract(point, point2).astype(np.float64)) < 40:
            distance = distance + 1
    return distance


def build_interpretation(primitives, lines, circles, intersections, text_regions, image_shape, detect_labels):
    primitive_list = [primitive.coords for primitive in primitives]
    # PARAM diagram_graph_builder_clustering_eps
    dbscan_eps = Params.params['diagram_graph_builder_dbscan_eps']
    clustering = DBSCAN(eps=dbscan_eps * (image_shape[0] + image_shape[1]) / 2, min_samples=1).fit(primitive_list)
    cluster_list = []
    num_clusters = max(clustering.labels_) + 1
    for _ in range(num_clusters):
        cluster_list.append(PrimitiveGroup())
    for idx, label in enumerate(clustering.labels_):
        cluster_list[label].add(primitives[idx])
    if detect_labels:
        character_predictor = CharacterPredictor()

        upper = []
        numbers = []
        for idx, coords in enumerate(text_regions.keys()):

            isupper, confidence = character_predictor.is_upper(text_regions[coords])
            if isupper:
                upper.append((Primitive(coords, 't', idx, character_type='upper'), confidence))
            else:
                numbers.append(Primitive(coords, 't', idx, character_type='upper'))
            primitives.append(Primitive(coords, 't', idx, character_type='upper'))
        sorted_upper = [item[0] for item in sorted(upper, key=lambda item: item[1], reverse=True)]
        upper = sorted_upper
        search_queue = Queue()
        search_queue.put(SearchNode(primitives, lines, cluster_list))
        best_child = SearchNode(primitives, lines, cluster_list)
        offset_factor = Params.params['primitive_group_weight_offset_factor']

        while search_queue and upper:
            node = search_queue.get()
            if node.level >= len(upper):
                break
            children = node.generate_children(upper[node.level])
            sorted_children = sorted(children, key=lambda x: x.fitness(
                weight_offset=offset_factor * (image_shape[0] + image_shape[1]) / 2))
            best_child = sorted_children[-1]

            if len(sorted_children) < 1:
                for child in sorted_children:
                    search_queue.put(child)
            else:
                for child in sorted_children[-1:]:
                    search_queue.put(child)
    else:
        best_child = SearchNode(primitives, lines, cluster_list)

    diagram_interpretation = Interpretation()
    # for cluster in best_child.points:
    #     if cluster.contains('t'):

    for primitive_group_index, cluster in enumerate(best_child.points):
        point = Point()
        if cluster.contains('i'):
            point.set_coords(cluster.centroid('i'))
            for i in cluster.primitives['i']:
                structures = list(intersections.values())[i.index]
                for structure in structures:
                    if str.endswith(structure, 'center'):
                        # remove '_center' from the string
                        point.add_property('centerof', structure[0:len(structure) - 7])
                    else:
                        point.add_property('lieson', structure)

        if cluster.contains('c'):
            if point.coords is None:
                point.set_coords(cluster.centroid('c'))
            line_set = set()
            corner_centroid = cluster.centroid('c')
            eps = Params.params['diagram_parser_corner_lies_on_line_eps']

            for index, line in enumerate(lines):
                # PARAM: diagram_parser_corner_lies_on_line_eps
                if point_to_line_distance(corner_centroid, line) < eps * (image_shape[0] + image_shape[1]) / 2:
                    line_set.add(index)
            for line in line_set:
                point.add_property('lieson', f'l{line}')
            # circle_set = set()
            # for index, circle in enumerate(circles):
            #     if point_to_circle_distance(corner_centroid, circle) < eps * (image_shape[0] + image_shape[1]) / 2:
            #         circle_set.add(index)
            # for index in circle_set:
            #     point.add_property('lieson', f'c{index}')

        if cluster.contains('t'):
            for text_primitive in cluster.primitives['t']:
                text_region = list(text_regions.values())[text_primitive.index]
                text = ''.join(
                    [character_predictor.predict_character(character_region, character_mode='letters') for
                     character_region in text_region]).upper()

                point.set_label(text)
        else:
            label = f'p{diagram_interpretation.num_points()}'
            point.set_label(label)
        if len(point.properties) != 0 and (len(circles) == 0 or cluster.contains('i') or cluster.contains('t')):
            diagram_interpretation.add_point(point)
    # point_projections = get_point_projections(lines, diagram_interpretation)
    # number_search_queue = Queue()
    # best_child.reset_level()
    # best_child.set_point_projections(point_projections)
    # number_search_queue.put(best_child)
    # best_child_with_numbers = best_child
    # while number_search_queue:
    #     node = number_search_queue.get()
    #     if node.level >= len(numbers):
    #         break
    #     children = node.generate_children(numbers[node.level])
    #     sorted_children = sorted(children, key=lambda x: x.fitness())
    #     best_child_with_numbers = sorted_children[-1]
    #
    #     if len(sorted_children) < 1:
    #         for child in sorted_children:
    #             number_search_queue.put(child)
    #     else:
    #         for child in sorted_children[-1:]:
    #             number_search_queue.put(child)
    # diagram_interpretation.set_info_labels(best_child_with_numbers.info_labels)
    diagram_interpretation.set_lines(lines)
    diagram_interpretation.set_circles(circles)
    return diagram_interpretation


def build_graph_from_interpretation(interpretation):
    graph = nx.Graph()
    for point in interpretation:
        graph.add_node(point.label)
    line_points = dict()
    for key, line in interpretation.lines.items():
        points_on_line = []
        for point in interpretation.points:
            if point.has_property(('lieson', key)):
                points_on_line.append(point)
        if np.pi / 4 < inclination(line[1]) < 3 * np.pi / 4:
            points_on_line = sorted(points_on_line, key=lambda p: p.coords[1])
        else:
            points_on_line = sorted(points_on_line, key=lambda p: p.coords[0])
        line_points[key] = points_on_line

    for (point1, point2) in itertools.combinations(interpretation.points, 2):
        if len(point1.properties.intersection(point2.properties)):
            data = []
            for common_property in point1.properties.intersection(point2.properties):
                if common_property[0] == 'lieson' and common_property[1][0] == 'l':
                    points = line_points[common_property[1]]
                    idx1 = points.index(point1)
                    idx2 = points.index(point2)
                    data.extend(points[min(idx1, idx2) + 1:max(idx1, idx2)])
            graph.add_edge(point1.label, point2.label, data=data)

    return graph


def get_point_projections(lines, interpretation):
    points_on_line = dict()
    for point in interpretation.points:
        for property in point.properties:
            if property[0] == 'lieson' and property[1][0] == 'l':
                if property[1] not in points_on_line:

                    points_on_line[property[1]] = [
                        (point.label[0], point.coords)]
                else:
                    points_on_line[property[1]].append(
                        (point.label[0], point.coords))
    for line_index, points in points_on_line.items():
        sk_line = find_sk_line(lines[int(line_index[1])])
        projected_points = []
        for point in points:
            projected_coords = sk_line.project_point(skobj.Point(point[1]))
            projected_points.append((point[0], projected_coords))
        points_on_line[line_index] = projected_points
    for line_index, points in points_on_line.items():
        line = lines[int(line_index[1])]
        if np.pi / 4 < inclination(line[1]) < 3 * np.pi / 4:
            points_on_line[line_index] = sorted(points, key=lambda p: p[1][1])
        else:
            points_on_line[line_index] = sorted(points, key=lambda p: p[1][0])
    return points_on_line


def find_sk_line(hesse_line):
    rho = hesse_line[0]
    theta = hesse_line[1]
    point = (rho * cos(theta), rho * sin(theta))
    rotation_matrix = np.array([[cos(np.pi / 2), -sin(np.pi / 2)],
                                [sin(np.pi / 2), cos(np.pi / 2)]])
    if point != (0., 0.):
        direction = np.dot(rotation_matrix, point)
    else:
        direction = (cos(theta), sin(theta))

    return skobj.Line(point, direction)


def display_interpretation(image, interpretation, lines, circles):
    for idx, point in enumerate(interpretation):
        hue = 179 * idx / len(interpretation.points)
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        int_coords = (int(point.coords[0]), int(point.coords[1]))

        cv2.circle(image, int_coords, 2, rgb.tolist(), -1)

        cv2.putText(image, point.label, int_coords, cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 0, 0))
    line_img = draw_lines(image, lines)
    circle_img = draw_circles(image, circles)
    cv2.imshow('lines', line_img)
    cv2.imshow('circles', circle_img)
    cv2.imshow('interpretation', image)
    cv2.waitKey()

# diagram = cv2.imread('test_images/Untitled.png')
# interpretation, lines, circles = parse_diagram(diagram)
# print(interpretation)
# display_interpretation(diagram, interpretation, lines.values(), circles.values())
# cv2.destroyAllWindows()
# import os
# import time
#
# count = 0
# selecting = 0
# totalstart = time.time()
# for filename in os.listdir('../experiments/data/images'):
#     if filename.endswith('.png') and len(filename) == 8:
#         try:
#             diagram = cv2.imread('../experiments/data/images/' + filename)
#
#             interpretation, lines, circles = parse_diagram(diagram)
#             display_interpretation(diagram, interpretation, lines.values(), circles.values())
#             stop = time.time()
#             print(time.time() - totalstart)
#             count += 1
#             print(filename)
#             print(f'files done: {count}\r')
#         except IndexError:
#             pass
#         except ValueError:
#             pass
#
# totalstop = time.time()
# print(totalstop - totalstart)
# print(selecting)
