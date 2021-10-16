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
    def coord_list(self, type=None, labels_only=None, info_text_only=None):
        coord_list = list()
        for key, value in self.primitives.items():
            if type is None:
                coord_list.extend([primitive.coords for primitive in value])
            elif key == type:
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
    def centroid(self, *type_list):
        if len(type_list) == 0:
            return np.mean(self.coord_list(), axis=0)
        else:
            coords = []
            for type in type_list:
                if type in self.primitives:
                    primitives_with_type = self.primitives[type]
                    coords.extend([primitive.coords for primitive in primitives_with_type])
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

    '''
    calculates score for this primitivegroup (higher is better)
    :param offset: neutral point.
    :return:
    '''
    def weight(self, offset=0):
        if len(self.primitives['t']) > 1:
            return -9999
        weight = 0
        if self.contains('i'):
            i_centroid = self.centroid('i')
            for text_coord in self.coord_list('t'):
                weight += np.linalg.norm(text_coord - i_centroid)
        elif self.contains('c'):
            c_centroid = self.centroid('c')
            for text_coord in self.coord_list('t', labels_only=True):
                weight += np.linalg.norm(text_coord - c_centroid)
        weight = weight / len(self.coord_list('t'))
        return offset-weight
    def __str__(self):
        return f'PrimitiveGroup({self.primitives})'
    def __repr__(self):
        return  self.__str__()
    def __copy__(self):
        return copy.deepcopy(self)


