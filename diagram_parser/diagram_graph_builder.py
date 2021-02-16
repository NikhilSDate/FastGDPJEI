from diagram_parser.line_detecter import get_filtered_lines, draw_lines
from diagram_parser.corner_detector import get_corners, draw_corners
import cv2.cv2 as cv2
import numpy as np

image = cv2.imread('../aaai/000.png')
lines = get_filtered_lines(image)
image_with_lines = draw_lines(image, lines)
cv2.imshow('image', image_with_lines)
cv2.waitKey()

corners=get_corners(image)
image_with_corners=draw_corners(image, corners)
cv2.imshow('image with corners', image_with_corners)
cv2.waitKey()

def get_intersection(line1, line2):
    pass

