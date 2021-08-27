import numpy as np
from diagram_parser.primitivegroup import PrimitiveGroup
from diagram_parser.info_label import InfoLabel
import copy
import math


class SearchNode:

    def __init__(self, primitives, lines, points=None, point_projections=None):
        if points is None:
            self.points = list()
        else:
            self.points = points
        self.info_labels = list()
        self.noise_set = set(primitives)
        self.primitive_set = set(primitives)
        self.lines = lines
        self.level = 0
        self.point_projections = point_projections
    def generate_children(self, primitive):
        children = []
        possible_operations = ['do_nothing']
        child = self.__copy__()
        child.level = self.level + 1
        children.append(child)
        if primitive.is_letter:
            for idx, point in enumerate(self.points):
                possible_operations.append(('add_to_point', point))
                child1 = self.__copy__()
                child1.points[idx].add(primitive)
                child1.noise_set.remove(primitive)
                child1.level = self.level + 1

                children.append(child1)
        else:
            for idx, point in enumerate(self.points):
                child1 = self.__copy__()
                child1.info_labels.append(InfoLabel(primitive, idx, 'a'))
                child1.noise_set.remove(primitive)
                child1.level = self.level + 1
                children.append(child1)
            for idx, line in enumerate(self.lines):
                child1 = self.__copy__()
                child1.info_labels.append(InfoLabel(primitive, idx, 'l'))
                child1.noise_set.remove(primitive)
                child1.level = self.level + 1
                children.append(child1)
        return children

    def fitness(self, weight_offset=0):
        fitness = 0
        for point in self.points:
            if point.contains('t'):
                fitness += point.weight(weight_offset)
        for info_label in self.info_labels:
            fitness += info_label.weight(self.points, self.lines, self.point_projections)
        return fitness

    def total_variance(self):
        coord_set = [primitive.coords for primitive in self.primitive_set]
        data = np.transpose(np.array(coord_set))
        return np.trace(np.cov(data))
    def reset_level(self):
        self.level = 0
    def set_point_projections(self, projections):
        self.point_projections = projections
    def __copy__(self):
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return f'SearchNode(points = {self.points},\n Info groups = {self.info_labels})'

    def __repr__(self):
        return '\n' + self.__str__()
