from sympy import *
import euclid
class Statement:

    def __init__(self, statement_string):
        self.points = list()
        tokens=statement_string.split(' ')
        if len(tokens) == 3:
            self.type = tokens[0]
            points = tokens[1]
            self.value = float(tokens[2])
            for point in points:
                self.points.append(point)
    @staticmethod
    def get_symbols(point_name):
        x=Symbol('x'+point_name)
        y=Symbol('y'+point_name)
        return x, y

    def get_statement_symbols(self):
        symbols = set()
        for point in self.points:
            x,y=Statement.get_symbols(point)
            symbols.add(x)
            symbols.add(y)
        return symbols

    def get_equation(self):
        if self.is_question():
            if self.type[0,-1]=='length':
                pass
        else:
            if self.type == 'length':
                print(self.points)
                p1 = Statement.get_symbols(self.points[0])
                p2 = Statement.get_symbols(self.points[1])
                return euclid.length_equation(p1,p2,length=self.value)
            if self.type == 'angle':
                p1 = Statement.get_symbols(self.points[0])
                p2 = Statement.get_symbols(self.points[1])
                p3 = Statement.get_symbols(self.points[2])
                return euclid.angle_equation(p1,p3,p2,theta=self.value)
    def is_question(self):
        return self.type.endswith('?')


statement=Statement('length XY 3.1415')
pprint(statement.get_equation())
string='abcde'
print(string[0:-1])







