import numpy as np
import copy
import math
class PrimitiveGroup:
    def __init__(self, cluster_index=None):
        self.primitives = dict()
        self.cluster_index = cluster_index
        self.num_primitives = 0
    def add(self, primitive):
        if primitive.type in self.primitives:
            self.primitives[primitive.type].append(primitive)
        else:
            self.primitives[primitive.type] = [primitive]
        self.num_primitives = self.num_primitives + 1
    def contains(self, type):
        return type in self.primitives
    def coord_list(self):
        coord_list = list()
        for key, value in self.primitives.items():
            coord_list.extend([primitive.coords for primitive in value])
        return coord_list
    def variance(self):
        if len(self.coord_list())<2:
            return 0
        else:
            data = np.transpose(np.array(self.coord_list()))
            return np.trace(np.cov(data))
    def diameter(self):
        diameter = 0
        for coord1 in self.coord_list():
            for coord2 in self.coord_list():
                if np.linalg.norm(np.subtract(coord2, coord1)) > diameter:
                    diameter = np.linalg.norm(np.subtract(coord2, coord1))
        return diameter
    def centroid(self, type=None):
        if type is None:
            return np.mean(self.coord_list(), axis=0)
        else:
            primitives_with_type = self.primitives[type]
            coords = [primitive.coords for primitive in primitives_with_type]
            return np.mean(coords, axis=0)
    def penalty(self):
        penalty = 0
        for type, primitive_list in self.primitives.items():
            if type == 't':
                if len(primitive_list) > 1:
                    penalty += 100 * (len(primitive_list) - 1)
            elif type == 'i':
                if len(primitive_list) > 1:
                    penalty += 50 * (len(primitive_list) - 1)
        if 'c' not in self.primitives and 'i' not in self.primitives:
            print('only text')
            penalty += 10000
        return penalty
    def weight(self):
        weight = 0
        if self.contains('i'):
            weight = np.linalg.norm(self.centroid('t') - self.centroid('i'))
        elif self.contains('c'):
            weight = np.linalg.norm(self.centroid('t') - self.centroid('c'))
        return 1/weight
    def __str__(self):
        return f'PrimitiveGroup({self.primitives})'
    def __repr__(self):
        return  self.__str__()
    def __copy__(self):
        return copy.deepcopy(self)


