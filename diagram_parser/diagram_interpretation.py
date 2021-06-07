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
        return f'Interpretation(points={self.points}, info={self.info_labels})'
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
    def from_xml():
        pass
    def __getitem__(self, item):
        return self.points[item]

