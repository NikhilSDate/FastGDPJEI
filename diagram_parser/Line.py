from numpy import cos, sin, tan , sqrt, arctan2
class Line:
    def __init__(self, id, vals, value_format='hesse_normal'):
        self.id = id
        self.value_format = value_format
        self.vals = vals
    def get_vals(self, target_format='two_point'):
        if target_format == 'two_point':
            if self.value_format == 'two_point':
                return self.vals
            elif self.value_format == 'hesse_normal':
                raise Exception()
        if target_format == 'hesse_normal':
            if self.value_format == 'two_point':
                x1, y1, x2, y2 = self.vals
                A = y1 - y2
                B = x2 - x1
                C = (x1 - x2) * y1 + (y2 - y1) * x1
                cosine = A / sqrt(A ** 2 + B ** 2)
                sine = B / sqrt(A ** 2 + B ** 2)
                negative_rho = C / sqrt(A ** 2 + B ** 2)
                rho = -negative_rho
                theta = arctan2(sine, cosine)
                return [rho, theta]
            elif self.value_format == 'hesse_normal':
                return self.vals