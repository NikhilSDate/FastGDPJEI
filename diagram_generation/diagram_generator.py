import math
from cairo import ImageSurface, Context
import cairo
import random

def generate_diagram_with_line():
    surface = ImageSurface(cairo.FORMAT_RGB24, 200, 200)
    ctx = Context(surface)
    ctx.save()
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.restore()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(2)
    num_lines = 0
    lattice = set()
    while num_lines < 5:
        if len(lattice) == 0:
            point1 = (random.random()*200, random.random()*200)
            point2 = (random.random()*200, random.random()*200)
            ctx.move_to(*point1)
            ctx.line_to(*point2)
            ctx.stroke()
            lattice.update(get_line_points(point1, point2))
            num_lines+=1
        else:
            point1 = random.sample(lattice, 1)[0]
            point2 = random.sample(lattice, 1)[0]
            ctx.move_to(*point1)
            ctx.line_to(*point2)
            ctx.stroke()
            lattice.update(get_line_points(point1, point2))
            num_lines+=1
    surface.write_to_png('test.png')

def get_line_points(point1, point2):
    mid = tuple([sum(x)/2 for x in zip(point1, point2)])
    return [point1, mid, point2]
generate_diagram_with_line()




