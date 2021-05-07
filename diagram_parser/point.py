class Point():
    def __init__(self):
        self.labels = list()
        self.properties = set()
    def add_label(self, label=None):
        self.labels.append(label)
    def add_property(self, property_name, property_data):
        self.properties.add((property_name, property_data))
    def __str__(self):
        return f'Point(labels={self.labels}, properties={self.properties})'
    def __repr__(self):
        return self.__str__()