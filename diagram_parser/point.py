class Point():
    def __init__(self):
        self.labels = list()
        self.properties = set()
        self.coords = None
    def add_label(self, label=None):
        self.labels.append(label)
    def set_coords(self, coords):
        self.coords = coords
    def add_property(self, property_name, property_data):
        self.properties.add((property_name, property_data))
    def has_property(self, property):
        return property in self.properties
    def __str__(self):
        return f'Point(labels={self.labels}, properties={self.properties}, coords={self.coords})'
    def __repr__(self):
        return self.__str__()
    def num_properties(self):
        return len(self.properties)
    def scale_coords(self, scale_factor):

        self.coords = [self.coords[0]*scale_factor, self.coords[1]*scale_factor]