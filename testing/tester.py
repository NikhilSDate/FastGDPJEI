from diagram_parser.diagram_interpretation import Interpretation
from diagram_parser.line_detecter import close_enough, hesse_normal_form
from diagram_parser.point import Point
from diagram_parser.circle_detector import circles_close_enough
from testing.params import Params
import numpy as np
import xml.etree.ElementTree as ET
from diagram_parser.diagram_graph_builder import parse_diagran
import cv2.cv2 as cv2



def distance(point1_coords, point2_coords):
    return np.linalg.norm(point1_coords - point2_coords)


def labels_match(ground_truth_label, predicted_label):
    if (ground_truth_label[0] == 'P' and len(ground_truth_label) > 1) and (
            predicted_label[0] == 'p' and len(predicted_label) > 1):
        return True
    elif ground_truth_label == predicted_label:
        return True


def f1_score(ground_truth_interpretation, ground_truth_lines, ground_truth_circles, predicted_interpretation,
             predicted_lines, predicted_circles):
    matched_points = [False] * len(predicted_interpretation.points)
    point_match = dict()
    DISTANCE_THRESHOLD = 10
    for idx1, point1 in enumerate(ground_truth_interpretation):
        for idx2, point2 in enumerate(predicted_interpretation):
            if not matched_points[idx2] and distance(point1.coords,
                                                     point2.coords) < DISTANCE_THRESHOLD and labels_match(
                point1.labels[0], point2.labels[0]):
                matched_points[idx2] = True
                point_match[idx1] = idx2
    line_and_circle_match = dict()
    matched_lines = set()
    for id1, line in ground_truth_lines.items():
        for id2, predicted_line in predicted_lines.items():
            if id2 not in matched_lines and close_enough(
                    hesse_normal_form((line[0][0], line[0][1], line[1][0], line[1][1])), predicted_line,
                    image_size=(200, 200)):
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


def parse_annotations():
    tree = ET.parse('../aaai/annotations_2.xml')
    for child in tree.getroot():
        if child.tag == 'image':
            file_name = child.attrib['name']
            annotations = [annotation for annotation in child]
            ground_truth_lines = dict()
            ground_truth_circles = dict()
            diagram_interpretation = Interpretation()
            for annotation in annotations:
                if annotation.attrib['label'] == 'Point':
                    current_point = Point()
                    coords = [float(coord) for coord in annotation.attrib['points'].split(',')]
                    current_point.set_coords(coords)
                    for annotation_child in annotation:
                        if annotation_child.attrib['name'] == 'id':
                            current_point.add_label(annotation_child.text)
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
                    ground_truth_lines[annotation[0].text] = np.array((point1, point2))
                elif annotation.attrib['label'] == 'Circle':
                    circle_points = annotation.attrib['points'].split(';')
                    point1 = [float(coord) for coord in circle_points[0].split(',')]
                    point2 = [float(coord) for coord in circle_points[1].split(',')]
                    ground_truth_circles[annotation[0].text] = (point1, point2)
            processed_circles = dict()
            for key, circle in ground_truth_circles.items():
                center = circle[0]
                radius = np.linalg.norm(np.subtract(circle[1], circle[0]))
                processed_circles[key] = np.array((*center, radius))
            diagram_interpretation.set_lines(ground_truth_lines)
            diagram_interpretation.set_circles(ground_truth_circles)
            yield file_name, diagram_interpretation, ground_truth_lines, processed_circles


def run_test():
    total_relevant_properties = 0
    total_predicted_properties = 0
    total_ground_truth_properties = 0
    f1_scores = []
    count = 0
    for file_name, interpretation, lines, circles in parse_annotations():
        if interpretation.total_properties() > 0:

            diagram_image = cv2.imread(f'../aaai/{file_name}')
            predicted_interpretation, predicted_lines, predicted_circles = parse_diagran(diagram_image)
            # print(predicted_interpretation)
            f1_info = f1_score(interpretation, lines, circles, predicted_interpretation, predicted_lines,
                               predicted_circles)
            total_relevant_properties += f1_info[0]
            total_predicted_properties += f1_info[1]
            total_ground_truth_properties += f1_info[2]
            diagram_score = \
                f1_score(interpretation, lines, circles, predicted_interpretation, predicted_lines, predicted_circles)[
                    3]
            f1_scores.append(diagram_score)
            count += 1
            print(f'files done:{count}')
            if file_name == '042.png':
                break
    total_precision = total_relevant_properties / total_predicted_properties
    total_recall = total_relevant_properties / total_ground_truth_properties

    return f1_scores, total_precision, total_recall
