from diagram_parser.diagram_interpretation import Interpretation
from diagram_parser.line_detecter import close_enough, hesse_normal_form
from diagram_parser.point import Point
import numpy as np
import xml.etree.ElementTree as ET
from diagram_parser.diagram_graph_builder import parse_diagran
import cv2.cv2 as cv2
def distance(point1_coords, point2_coords):
    return np.linalg.norm(point1_coords - point2_coords)


def labels_match(ground_truth_label, predicted_label):
    if (ground_truth_label[0] == 'p' and len(ground_truth_label) == '2') and (
            predicted_label[0] == 'p' and len(predicted_label) == '2'):
        return True
    elif ground_truth_label == predicted_label:
        return True


def f1_score(ground_truth_interpretation, ground_truth_lines_and_circles, predicted_interpretation, predicted_lines_and_circles):
    matched_points = [False] * len(predicted_interpretation.points)
    point_match = dict()
    DISTANCE_THRESHOLD = 5
    for idx1, point1 in enumerate(ground_truth_interpretation):
        for idx2, point2 in enumerate(predicted_interpretation):
            if not matched_points[idx2] and distance(point1.coords, point2.coords) < DISTANCE_THRESHOLD and labels_match(
                    point1.labels[0], point2.labels[0]):
                matched_points[idx2] = True
                point_match[idx1] = idx2
    line_circle_match = dict()
    matched_lines = set()
    for id1, line in ground_truth_lines_and_circles:
        for id2, predicted_line in predicted_lines_and_circles:
            if id2 not in matched_lines and close_enough(hesse_normal_form(line), predicted_line, image_size=(200, 200)):
                matched_lines.add(id2)
                line_circle_match[id1] = id2
    num_relevant_properties = 0
    predicted_point_list = list(predicted_interpretation)
    for ground_truth_idx, ground_truth_point in enumerate(ground_truth_interpretation):
        matched_point = predicted_point_list[point_match[ground_truth_idx]]
        for ground_truth_property in ground_truth_point:
            matched_property = (ground_truth_property[0], line_circle_match[ground_truth_property[1]])
            if matched_point.has_property(matched_property):
                num_relevant_properties += 1
    precision = num_relevant_properties / predicted_interpretation.total_properties()
    recall = num_relevant_properties / ground_truth_interpretation.total_properties()
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def parse_annotations():
    tree = ET.parse('../annotations.xml')
    for child in tree.getroot():
        if child.tag == 'image':
            annotations = [annotation for annotation in child]
            lines_and_circles = dict()

            diagram_interpretation = Interpretation()
            for annotation in annotations:
                if annotation.attrib['label'] == 'Point':
                    current_point = Point()
                    coords = [float(coord) for coord in annotation.attrib['points'].split(',')]
                    current_point.set_coords(coords)
                    for annotation_child in annotation:
                        if annotation_child.attrib['name'] == 'id':
                            current_point.add_label(annotation_child.text)
                        elif annotation_child.attrib['name'] == 'lieson':
                            if annotation_child.text is not None:

                                line_circle_list = [x.strip() for x in annotation_child.text.split(',')]
                                for line_or_circle in line_circle_list:
                                    current_point.add_property('lieson', line_or_circle)
                        elif annotation_child.attrib['name'] == 'centerof':
                            if annotation_child.text is not None:
                                circles = [x.strip() for x in annotation_child.text.split(',')]
                                for circle in circles:
                                    current_point.add_property('centerof', circle)
                    diagram_interpretation.add_point(current_point)
                elif annotation.attrib['label'] == 'Line':
                    line_points = annotation.attrib['points'].split(';')
                    point1 = [float(coord) for coord in line_points[0].split(',')]
                    point2 = [float(coord) for coord in line_points[1].split(',')]
                    lines_and_circles[annotation[0].text] = (point1, point2)
                elif annotation.attrib['label'] == 'Circle':
                    circle_points = annotation.attrib['points'].split(';')
                    point1 = [float(coord) for coord in circle_points[0].split(',')]
                    point2 = [float(coord) for coord in circle_points[1].split(',')]
                    lines_and_circles[annotation[0].text] = (point1, point2)
            yield diagram_interpretation, lines_and_circles

diagram_image = cv2.imread('../aaai/000.png')
predicted_interpretation, lines, circles = parse_diagran(diagram_image)
print(predicted_interpretation)
for interpretation, lines_and_circles in parse_annotations():
    if interpretation.total_properties()>0:
        print(f1_score(interpretation, lines_and_circles, predicted_interpretation, (lines, circles)))
    # for element in child:
    #     print(element.tag, element.attrib)
    #     for attribute in element:
    #         print(attribute.text)
