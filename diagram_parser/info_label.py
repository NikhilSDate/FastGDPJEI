import numpy as np
from numpy import cos, sin, sqrt, exp, pi, inf
import skspatial.objects as skobj
from diagram_parser.line_detecter import inclination
class InfoLabel:
    # allowed types are 'a' (angle) and 'l' (length)
    def __init__(self, label, primitive_id, type):
        self.label = label
        self.primitive_id = primitive_id
        self.type = type
        self.sk_line = None
        self.boundary_points = []
    def coords(self):
        return self.label.coords
    def set_coords(self, coords):
        self.label.coords = coords

    def weight(self, primitive_groups, lines, point_projections):
        if self.type == 'l':
            # determines distance from midpoint of nearest two points on the line

            line = lines[self.primitive_id]
            points_on_line = point_projections[f'l{self.primitive_id}']
            if np.pi/4 < inclination(line[1]) < 3 * np.pi/4:
                axis_to_search = 1
            else:
                axis_to_search = 0
            i = 0
            while self.coords()[axis_to_search] > points_on_line[i][1][axis_to_search]:
                i = i + 1
                if i == len(points_on_line):
                    i = -1
                    break
            if i == 0 or i == -1:
                distance = inf
            else:
                line_midpoint = (points_on_line[i-1][1] + points_on_line[i][1])/2
                distance = line_midpoint.distance_point(self.coords())
                self.boundary_points = [points_on_line[i-1][0], points_on_line[i][0]]
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
        return f'InfoLabel({self.label}, {self.primitive_id}, {self.type}, {self.boundary_points})'
    def __repr__(self):
        return self.__str__()