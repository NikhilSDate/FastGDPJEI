import numpy as np
from diagram_parser.primitivegroup import PrimitiveGroup
import copy
import math


class SearchNode:

    def __init__(self, primitives, points=None):
        if points is None:
            self.points = list()
        else:
            self.points = points
        self.info_groups = list()
        self.noise_set = set(primitives)
        self.primitive_set = set(primitives)
        self.operations = ['new_point', 'add_to_point', 'add_to_point_duplicate',
                           'new_info_group', 'add_to_info_group', 'add_to_info_group_duplicate'
                                                                  'keep_as_noice']
        self.level = 0

    def generate_children(self, primitive):
        children = []
        possible_operations = []

        for idx, point in enumerate(self.points):
            possible_operations.append(('add_to_point', point))
            child1 = self.__copy__()
            child1.points[idx].add(primitive)
            child1.noise_set.remove(primitive)
            child1.level = self.level + 1

            children.append(child1)

        # possible_operations.append('keep_as_noise')
        # children.append(self.__copy__())
        return children

    def fitness(self):
        fitness = 0
        for point in self.points:
            if point.contains('t'):
                fitness += point.weight()


        # for noise_point in self.noise_set:
        #     variance_before = self.total_variance()
        #     self.primitive_set.remove(noise_point)
        #     variance_after = self.total_variance()
        #     self.primitive_set.add(noise_point)
        #     fitness = fitness + (len(self.noise_set)/len(self.primitive_set))*(variance_before - variance_after) / variance_before
        return fitness

    def total_variance(self):
        coord_set = [primitive.coords for primitive in self.primitive_set]
        data = np.transpose(np.array(coord_set))
        return np.trace(np.cov(data))

    def __copy__(self):
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return f'SearchNode(points = {self.points},\n Info groups = {self.info_groups}, \nNoise = {self.noise_set})'

    def __repr__(self):
        return '\n' + self.__str__()
