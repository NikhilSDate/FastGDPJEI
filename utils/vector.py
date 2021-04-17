import cairo
import cv2.cv2 as cv2
import numpy as np
def draw_vector_contours(contours, holes, image_size, scale=10):
    width = image_size[0]
    height = image_size[1]
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width*scale, height*scale)
    ctx = cairo.Context(surface)
    for contour in contours:
        pass
        print(contour[0][0][0], contour[0][0][1])
        ctx.move_to(contour[0][0][0]*scale, contour[0][0][1]*scale)
        for point in contour:
            print('point', point[0][0], point[0][1])
            ctx.line_to(point[0][0]*scale, point[0][1]*scale)

        ctx.set_source_rgb(255, 255, 255)
        ctx.fill()
    for contour in holes:
        ctx.move_to(contour[0][0][0]*scale, contour[0][0][1]*scale)
        for point in contour:
            ctx.line_to(point[0][0]*scale, point[0][1]*scale)
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
#    ctx.paint()
    return surface
# def resize(surface):
#     ctx = cairo.Context(surface)
#     ctx.scale(10, 10)
#     return surface
# surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 300, 200)
# ctx = cairo.Context(surface)
# ctx.rectangle(25, 30, 100, 100)
# ctx.set_source_rgb(1, 0, 0)
# ctx.fill()
# surface.write_to_png('test.png')
# array = np.ndarray(shape=(100, 100, 3), dtype=np.uint8, buffer=surface.get_data())
# cv2.imshow('array', array)
# cv2.waitKey()

