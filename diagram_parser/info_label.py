import numpy as np
from numpy import cos, sin, sqrt, exp, pi
import skspatial.objects as skobj

class InfoLabel:
    # allowed types are 'a' (angle) and 'l' (length)
    def __init__(self, label, primitive_id, type):
        self.label = label
        self.primitive_id = primitive_id
        self.type = type
        self.sk_line = None
    def coords(self):
        return self.label.coords

    def weight(self, point_projections):
        if self.type == 'l':
            distance = InfoLabel.point_to_line_distance(self.coords(), line)
            return exp(-distance)
        elif self.type == 'a':
            point = primitive_groups[self.primitive_id]
            # point = current_interpretation[self.primitive_id]
            point_centroid = None
            if point.contains('i'):
                point_centroid = point.centroid('i')
            elif point.contains('c'):
                point_centroid = point.centroid('c')
            distance = np.linalg.norm(self.coords()-point_centroid)
            return exp(-distance)
    @staticmethod
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
    @staticmethod
    def find_sk_line(hesse_line):
        rho = hesse_line[0]
        theta = hesse_line[1]
        point = (rho*cos(theta), rho*sin(theta))
        rotation_matrix = np.array([[cos(pi/2), -sin(pi/2)],
                                    [sin(pi/2), cos(pi/2)]])
        direction = np.dot(rotation_matrix, point)
        return skobj.Line(point, direction)
    def __str__(self):
        return f'InfoLabel({self.label}, {self.primitive_id}, {self.type})'
    def __repr__(self):
        return self.__str__()