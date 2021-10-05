class Interpretation:
    def __init__(self):
        self.points = list()
        self.info_labels = list()
        self.lines = dict()
        self.circles = dict()
    def to_dict(self):
        point_dict = dict()
        for point in self.points:
            point_dict[point.primitive_group_idx] = point
        return point_dict

    def add_point(self, point):
        self.points.append(point)

    def __str__(self):
        return f'Interpretation(\npoints={self.points}, \ninfo={self.info_labels}, \nlines={self.lines}, \ncircles={self.circles})'
    def __repr__(self):
        return self.__str__()

    def num_points(self):
        return len(self.points)

    def __iter__(self):
        yield from self.points

    def total_properties(self):
        total_properties = 0
        for point in self.points:
            total_properties += point.num_properties()
        return total_properties
    def set_info_labels(self, info_labels):
        self.info_labels = info_labels
    @staticmethod
    def from_xml(tree):
        pass
    def __getitem__(self, item):
        return self.points[item]
    def rescale_coords(self, scale_factor):
        '''
                scales all coords in the diagram interpretation by scale_factor. (used when)
                :param scale_factor:
                :return:
        '''
        for point in self.points:
            point.scale_coords(scale_factor)


    def set_lines(self, lines):
        self.lines = lines

    def set_circles(self, circles):
        self.circles = circles
    def __len__(self):
        return len(self.points)



