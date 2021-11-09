from diagram_parser.diagram_interpretation import Interpretation
from diagram_parser.line_detecter import match_close_enough, hesse_normal_form
from diagram_parser.point import Point
from diagram_parser.circle_detector import circles_close_enough, circles_IOU_close_enough
import numpy as np
import xml.etree.ElementTree as ET
from diagram_parser.diagram_graph_builder import parse_diagram, display_interpretation
import cv2.cv2 as cv2
import pickle


def distance(point1_coords, point2_coords):
    return np.linalg.norm(point1_coords - point2_coords)


def labels_match(ground_truth_label, predicted_label):
    if len(ground_truth_label) > 1 and len(predicted_label) > 1:
        return True
    elif ground_truth_label == predicted_label:
        return True


def primitive_f1_score(ground_truth_interpretation, ground_truth_lines, ground_truth_circles, predicted_interpretation,
                       predicted_lines, predicted_circles, image_size):
    matched_lines = set()
    for id1, line in ground_truth_lines.items():
        for id2, predicted_line in predicted_lines.items():

            if id2 not in matched_lines and match_close_enough(
                    hesse_normal_form((line[0][0], line[0][1], line[1][0], line[1][1])), predicted_line,
                    image_size=image_size):
                matched_lines.add(id2)
                break
    matched_circles = set()
    for id1, circle in ground_truth_circles.items():
        for id2, predicted_circle in predicted_circles.items():
            if id2 not in matched_circles and circles_IOU_close_enough(circle, predicted_circle):
                matched_circles.add(id2)
                break
    num_relevant_primitives = len(matched_circles) + len(matched_lines)
    num_predicted_primitives = len(predicted_circles) + len(predicted_lines)
    num_total_primitives = len(ground_truth_circles) + len(ground_truth_lines)
    try:
        precision = num_relevant_primitives / num_predicted_primitives
        recall = num_relevant_primitives / num_total_primitives

        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return num_relevant_primitives, num_predicted_primitives, num_total_primitives, f1


def point_only_f1(ground_truth_interpretation, ground_truth_lines, ground_truth_circles, predicted_interpretation,
                  predicted_lines, predicted_circles, image_size):
    matched_points = [False] * len(predicted_interpretation.points)
    point_match = dict()
    DISTANCE_THRESHOLD = 0.05 * (image_size[0] + image_size[1]) / 2
    for idx1, point1, in enumerate(ground_truth_interpretation):
        label = point1.label
        if False:
            count = 0
            matched_idx = None
            for idx2, point2 in enumerate(predicted_interpretation):
                if labels_match(label, point2.label):
                    count += 1
                    if not matched_points[idx2] and distance(point1.coords,
                                                             point2.coords) < DISTANCE_THRESHOLD:
                        matched_idx = idx2
            if count == 1 and matched_idx is not None:
                matched_points[matched_idx] = True
                point_match[idx1] = matched_idx
        else:
            for idx2, point2 in enumerate(predicted_interpretation):
                if not matched_points[idx2] and distance(point1.coords,
                                                         point2.coords) < DISTANCE_THRESHOLD:
                    matched_points[idx2] = True
                    point_match[idx1] = idx2
    relevant_points = len(point_match)
    predicted_points = len(predicted_interpretation)
    ground_truth_points = len(ground_truth_interpretation)
    try:
        precision = relevant_points / ground_truth_points
        recall = relevant_points / predicted_points
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return relevant_points, predicted_points, ground_truth_points, f1


def f1_score(ground_truth_interpretation, ground_truth_lines, ground_truth_circles, predicted_interpretation,
             predicted_lines, predicted_circles, image_size):
    matched_points = [False] * len(predicted_interpretation.points)
    point_match = dict()
    DISTANCE_THRESHOLD = 0.05 * (image_size[0] + image_size[1]) / 2
    for idx1, point1, in enumerate(ground_truth_interpretation):
        label = point1.label
        if len(label) == 1:
            count = 0
            matched_idx = None
            for idx2, point2 in enumerate(predicted_interpretation):
                if labels_match(label, point2.label):
                    count += 1
                    if not matched_points[idx2] and distance(point1.coords,
                                                             point2.coords) < DISTANCE_THRESHOLD:
                        matched_idx = idx2
            if count == 1 and matched_idx is not None:
                matched_points[matched_idx] = True
                point_match[idx1] = matched_idx
        else:
            for idx2, point2 in enumerate(predicted_interpretation):
                if not matched_points[idx2] and distance(point1.coords,
                                                         point2.coords) < DISTANCE_THRESHOLD \
                        and labels_match(point1.label, point2.label):
                    matched_points[idx2] = True
                    point_match[idx1] = idx2

    line_and_circle_match = dict()
    matched_lines = set()
    for id1, line in ground_truth_lines.items():
        for id2, predicted_line in predicted_lines.items():
            if id2 not in matched_lines and match_close_enough(
                    hesse_normal_form((line[0][0], line[0][1], line[1][0], line[1][1])), predicted_line,
                    image_size=image_size):
                matched_lines.add(id2)
                line_and_circle_match[id1] = id2
    matched_circles = set()
    for id1, circle in ground_truth_circles.items():
        for id2, predicted_circle in predicted_circles.items():
            if id2 not in matched_circles and circles_close_enough(circle, predicted_circle):
                matched_circles.add(id2)
                line_and_circle_match[id1] = id2
    num_relevant_properties = 0
    predicted_point_list = list(predicted_interpretation)
    for ground_truth_idx, ground_truth_point in enumerate(ground_truth_interpretation):
        if ground_truth_idx in point_match:
            matched_point = predicted_point_list[point_match[ground_truth_idx]]
            for ground_truth_property in ground_truth_point.properties:
                if ground_truth_property[1] in line_and_circle_match:
                    matched_property = (ground_truth_property[0], line_and_circle_match[ground_truth_property[1]])
                    if matched_point.has_property(matched_property):
                        num_relevant_properties += 1
    precision = num_relevant_properties / predicted_interpretation.total_properties()
    recall = num_relevant_properties / ground_truth_interpretation.total_properties()
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return num_relevant_properties, predicted_interpretation.total_properties(), ground_truth_interpretation.total_properties(), f1


def parse_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    for child in tree.getroot():
        if child.tag == 'image':
            file_name = child.attrib['name']
            annotations = [annotation for annotation in child]
            ground_truth_lines = dict()
            ground_truth_circles = dict()
            line_idx = 0
            circle_idx = 0
            diagram_interpretation = Interpretation()
            for annotation in annotations:
                if annotation.attrib['label'] == 'Point':
                    current_point = Point()
                    coords = [float(coord) for coord in annotation.attrib['points'].split(',')]
                    current_point.set_coords(coords)
                    for annotation_child in annotation:
                        if annotation_child.attrib['name'] == 'id':
                            current_point.set_label(annotation_child.text)
                        elif annotation_child.attrib['name'] == 'liesOn':
                            if annotation_child.text is not None:

                                line_circle_list = [x.strip() for x in annotation_child.text.split(',')]
                                for line_or_circle in line_circle_list:
                                    current_point.add_property('lieson', line_or_circle)
                        elif annotation_child.attrib['name'] == 'centerOf':
                            if annotation_child.text is not None:
                                circles = [x.strip() for x in annotation_child.text.split(',')]
                                for circle in circles:
                                    current_point.add_property('centerof', circle)
                    diagram_interpretation.add_point(current_point)
                elif annotation.attrib['label'] == 'Line':
                    line_points = annotation.attrib['points'].split(';')
                    point1 = [float(coord) for coord in line_points[0].split(',')]
                    point2 = [float(coord) for coord in line_points[1].split(',')]
                    if annotation[0]:
                        ground_truth_lines[annotation[0].text] = (point1, point2)
                    else:
                        ground_truth_lines['l' + str(line_idx)] = (point1, point2)
                        line_idx += 1
                elif annotation.attrib['label'] == 'Circle':
                    circle_points = annotation.attrib['points'].split(';')
                    point1 = [float(coord) for coord in circle_points[0].split(',')]
                    point2 = [float(coord) for coord in circle_points[1].split(',')]
                    if annotation[0]:
                        ground_truth_circles[annotation[0].text] = (point1, point2)
                    else:
                        ground_truth_circles['c' + str(circle_idx)] = (point1, point2)
                        circle_idx += 1

            processed_circles = dict()
            for key, circle in ground_truth_circles.items():
                center = circle[0]
                radius = np.linalg.norm(np.subtract(circle[1], circle[0]))
                processed_circles[key] = np.array((*center, radius))
            diagram_interpretation.set_lines(ground_truth_lines)
            diagram_interpretation.set_circles(ground_truth_circles)
            yield file_name, diagram_interpretation, ground_truth_lines, processed_circles


def run_test(image_directory, annotation_path, image_set):
    with open('geos/points_test.pickle', 'rb') as f:
        points = pickle.load(f)

    def build_interpretation(image_points):
        interpretation = Interpretation()
        for coords in image_points:
            point = Point()
            point.set_coords(np.array(coords))
            point.set_label('p0')
            interpretation.add_point(point)
        return interpretation, {}, {}

    total_relevant_properties = 0
    total_predicted_properties = 0
    total_ground_truth_properties = 0
    file_scores = {}
    count = 0
    for file_name, interpretation, lines, circles in parse_annotations(annotation_path):
        if len(interpretation.points) > 0 and (image_set is None or file_name in image_set):
            diagram_image = cv2.imread(f'{image_directory}/{file_name}')

            predicted_interpretation, predicted_lines, predicted_circles = parse_diagram(diagram_image)
            # predicted_interpretation, predicted_lines, predicted_circles = build_interpretation(points[file_name])
            # display_interpretation(diagram_image, predicted_interpretation, predicted_lines.values(), predicted_circles.values())

            f1_info = point_only_f1(interpretation, lines, circles, predicted_interpretation, predicted_lines,
                                    predicted_circles, diagram_image.shape)
            total_relevant_properties += f1_info[0]
            total_predicted_properties += f1_info[1]
            total_ground_truth_properties += f1_info[2]
            diagram_score = f1_info[3]

            print(f1_info)
            file_scores[file_name] = diagram_score
            count += 1
            print(f'files done {count}')
            print(file_name)

    try:
        print('var', np.var(list(file_scores.values())))
        print('macro f1', np.mean(list(file_scores.values())))

        total_precision = total_relevant_properties / total_predicted_properties
        total_recall = total_relevant_properties / total_ground_truth_properties
        return file_scores, total_precision, total_recall

    except ZeroDivisionError:
        return 0, 0, 0


def run_primitive_test(image_directory, annotation_path, image_set=None):
    with open('geos/primitives_test.pickle', 'rb') as f:
        primitives = pickle.load(f)

    def process_primitives(primitives):
        lines = {}
        circles = {}
        for idx, primitive in enumerate(primitives):
            if len(primitive) == 4:
                lines['l' + str(idx)] = hesse_normal_form(primitive)
            else:
                circles['c' + str(idx)] = primitive
        return Interpretation(), lines, circles

    total_relevant_properties = 0
    total_predicted_properties = 0
    total_ground_truth_properties = 0
    count = 0
    file_f1_scores = {}
    for file_name, interpretation, lines, circles in parse_annotations(annotation_path):
        if len(interpretation.points) > 0 and (image_set is None or file_name in image_set):
            diagram_image = cv2.imread(f'{image_directory}/{file_name}')
            # predicted_interpretation, predicted_lines, predicted_circles = process_primitives(primitives[file_name])
            predicted_interpretation, predicted_lines, predicted_circles = parse_diagram(diagram_image)
            # display_interpretation(diagram_image, predicted_interpretation, predicted_lines.values(),
            #                        predicted_circles.values())

            f1_info = primitive_f1_score(interpretation, lines, circles, predicted_interpretation, predicted_lines,
                                         predicted_circles, diagram_image.shape)
            total_relevant_properties += f1_info[0]
            total_predicted_properties += f1_info[1]
            total_ground_truth_properties += f1_info[2]
            file_f1_scores[file_name] = f1_info[3]
            count += 1
            print(f1_info)
            print(file_name)
            print(f'files done: {count} \r')
    total_precision = total_relevant_properties / total_predicted_properties
    total_recall = total_relevant_properties / total_ground_truth_properties
    print('macro f1', np.mean(list(file_f1_scores.values())))
    return total_precision, total_recall
