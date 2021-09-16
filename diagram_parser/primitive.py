import copy
class Primitive:
    def __init__(self, coords, type, index, character_type=None):
        self.coords = coords
        self.type = type
        self.index = index
        # character_type only for text
        self.character_type = character_type

    def __str__(self) -> str:
        return f'Primitive({self.coords}, {self.type}, {self.index}, {self.character_type})'
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return hash((self.coords, self.type, self.index))
    def __eq__(self, other):
        if isinstance(other, Primitive):
            return self.coords == other.coords and self.type == other.type and self.index == other.index
        else:
            return False
    def __copy__(self):
        return copy.deepcopy(self)