class Point():
    def __init__(self, primitive_group_index):
        self.labels = list()
        self.properties = set()
        self.primitive_group_idx = primitive_group_index
    def add_label(self, label=None):
        self.labels.append(label)
    def add_property(self, property_name, property_data):
        self.properties.add((property_name, property_data))
    def has_property(self, property):
        return property in self.properties
    def __str__(self):
        return f'Point(labels={self.labels}, properties={self.properties}, primtive_group_idx={self.primitive_group_idx})'
    def __repr__(self):
        return self.__str__()