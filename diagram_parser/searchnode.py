import numpy as np
class SearchNode:

    def __init__(self, primitives, distances):
        self.point_set = set()
        self.info_groups = set()
        self.noise_set = set(primitives)
        self.primitive_set = set(primitives)
        self.distances = distances
        self.operations = ['new_point', 'add_to_point', 'add_to_point_duplicate',
                  'new_info_group', 'add_to_info_group', 'add_to_info_group_duplicate'
                  'keep_as_noice']
    def generate_children(self, primitive):
        possible_operations = ['new_point']
        for point in self.point_set:
            if not point.contains(primitive.type):
                possible_operations.append(('add_to_point', point))
            else
                possible_operations.append(('add_to_point_duplicate', point))
        for info_group in self.info_groups:
            if not info_group.contains(primitive.type):
                possible_operations.append(('add_to_info_group', info_group))
            else
                possible_operations.append(('add_to_info_group', info_group))
        possible_operations.append('keep_as_noise')

    def fitness(self):
        fitness = 0
        for point in self.point_set:
            fitness =  fitness + (self.total_variance() - point.variance())
        for noise_point in self.noise_set:
            variance_before =self.total_variance()
            self.primitive_set.remove(noise_point)
            variance_after = self.total_variance()
            self.primitive_set.add(noise_point)
            fitness = fitness + variance_before - variance_after
        return fitness

    def total_variance(self):
        coord_set = [primitive.coords for primitive in self.primitive_set]
        return np.trace(np.cov(coord_set))