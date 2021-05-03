import numpy as np
class PrimitiveGroup:
    def __init__(self, cluster_index):
        self.primitives = dict()
        self.cluster_index = cluster_index
    def add(self, primitive, type):
        if type in self.primitives:
            self.primitives[type].append(primitive)
        else:
            self.primitives[type] = [primitive]
    def contains(self, type):
        return type in self.primitives
    def coord_list(self):
        coord_list = list()
        for key, value in self.primitives.items():
            coord_list.extend([primitive.coords for primitive in value])
        return coord_list
    def variance(self):
        return np.trace(np.cov(self.coord_list()))

