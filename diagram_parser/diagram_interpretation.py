class Interpretation:
    def __init__(self):
        self.points = list()
        self.info = list()
    def to_dict(self):
        point_dict = dict()
        for point in self.points:
            point_dict[point.primitive_group_idx] = point
        return point_dict
    def add_point(self, point):
        self.points.append(point)
    def __str__(self):
        return f'Interpretation(points={self.points}, info={self.info})'

    def num_points(self):
        return len(self.points)