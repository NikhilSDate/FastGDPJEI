import cv2.cv2 as cv2
import numpy as np
import math
from math import acos, sin, pi
from sklearn.cluster import DBSCAN
from experiments.params import Params


def resize(image, final_larger_dim, interpolation=cv2.INTER_LINEAR):
    larger_dim = max(image.shape[0], image.shape[1])
    factor = final_larger_dim / larger_dim
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)


def preprocess(image):
    # foreground is black so erode instead of dilating
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    blur = cv2.GaussianBlur(eroded, (3, 3), 1.5)
    return blur


def run_hough_trial(image, param2, min_radius):
    param_1 = Params.params["hough_circles_param_1"]

    trial_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, 1,
                                     param_1, param2=param2, minRadius=min_radius,
                                     maxRadius=int((image.shape[0] + image.shape[1]) / 4))
    return trial_circles


def get_best_params(image, objective_function):
    trial_results = []
    x = []
    y = []
    circle_counts = []
    max_circles = 0
    param_2_range = Params.params["hough_circles_param_2_range"]
    for param2 in range(param_2_range[0], param_2_range[1], 2):
        for min_radius in range(0, int((image.shape[0] + image.shape[1]) / 4), 2):
            trial_circles = run_hough_trial(image, param2, min_radius)

            if trial_circles is not None and not np.array_equal(trial_circles, [[50], [0], [0], [0]]):
                num_circles = trial_circles.shape[1]
                trial_results.append((param2, min_radius, num_circles))
                circle_counts.append(num_circles)
                x.append(param2)
                y.append(min_radius)

                if num_circles > max_circles:
                    max_circles = num_circles


            else:
                x.append(param2)
                y.append(min_radius)
                circle_counts.append(0)
    # PARAM: hough_circles_max_param_2
    comparator = lambda trial: objective_function(*trial, param_2_range[1], int((image.shape[0] + image.shape[1]) / 4),
                                                  max_circles)
    sorted_results = sorted(trial_results, key=comparator)
    params = sorted_results[-1]

    return params


def area_of_circles(circles, image_shape):
    # the list of circles is nested in another list to follow the output format of cv2.HoughCircles(_
    pixels_inside = 0
    for x in range(0, image_shape[0], 5):
        for y in range(0, image_shape[1], 5):
            inside = False
            for circle in circles[0]:
                if (x - circle[0]) ** 2 + (y - circle[1]) ** 2 <= circle[2] ** 2:
                    inside = True
                    break
            if inside:
                pixels_inside = pixels_inside + 1
    return pixels_inside * 25 / (image_shape[0] * image_shape[1])


def objective_function(param2, min_radius, num_circles, max_param_2, max_radius, max_circles):
    if num_circles == 0:
        return 0
    else:
        # PARAM: hough_circles_objective_function_param_2_term_shape
        param_2_term_shape = Params.params['circle_detector_objective_function_param_2_term_shape']
        param2_term = (1 - math.exp(-param_2_term_shape * (param2 / max_param_2))) / (1 - math.exp(-param_2_term_shape))
        min_radius_term = math.exp((-(min_radius / max_radius) ** 2))
        # PARAM: hough_circles_objective_function_min_radius_term_shape
        min_radius_term_shape = Params.params['circle_detector_objective_function_min_radius_term_shape']
        min_radius_term = (1 - (min_radius_term_shape[0] * (min_radius / max_radius - min_radius_term_shape[1]) ** 2))
        num_circles_term = 1 - math.exp(-5 * math.pow(num_circles / max_circles, 1 / 4))
        # PARAM: hough_circles_param_2_weight
        min_radius_weight = Params.params['circle_detector_min_radius_weight']
        return param2_term + min_radius_weight * min_radius_term


def show_circles(image, circles):
    for i in circles:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', image)
    cv2.waitKey()


def clustering_filter(circles, image_size):
    circle_centers = [(circle[0], circle[1]) for circle in circles[0]]
    # PARAM: hough_circles_clustering_epsilon(/2)
    eps = Params.params['circle_detector_clustering_epsilon']
    clustering = DBSCAN(eps=eps * (image_size[0] + image_size[1]) / 2, min_samples=1).fit(circle_centers)
    clusters = dict()
    filtered_circles = []
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = [circles[0][idx]]
        else:
            clusters[label].append(circles[0][idx])
    for key in clusters.keys():
        average_circle = np.mean(clusters[key], axis=0)
        filtered_circles.append(average_circle)

    return np.array(filtered_circles)


def detect_circles(image):
    image = preprocess(image)
    max_dimension = max(image.shape[0], image.shape[1])
    resize_image = Params.params['resize_image_if_too_big']

    if max_dimension > 250 and resize_image:
        factor = 250 / max_dimension
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    else:
        factor = 1
    best_params = get_best_params(image, objective_function=objective_function)
    # TODO: MAYBE USE A MORE SOPHISTICATED WAY OF DETERMINING IF THE IMAGE CONTAINS CIRCLES. POSSIBLY OPTIMIZE
    #  THRESHOLD BY USING the area under the ROC
    # PARAM: hough_circles_is_a_circle_threshold
    is_a_circle = Params.params['circle_detector_is_a_circle_threshold']
    if best_params[0] > is_a_circle:

        best_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,
                                        1.5, 1, param1=50, param2=best_params[0],
                                        minRadius=best_params[1], maxRadius=int((image.shape[0] + image.shape[1]) / 4))
        # this code is here to catch a weird error
        if best_circles is not None:

            filtered_circles = clustering_filter(best_circles, image.shape)
        else:
            filtered_circles = None
        if filtered_circles is not None:
            return np.multiply(filtered_circles, 1 / factor)
        else:
            return np.array([])
    else:
        return np.array([])


# img = cv2.imread('../validation/images/0035.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# circles = detect_circles(gray)
# if circles is not None:
#     show_circles(img, circles.astype(np.uint16))


# ax = fig.gca(projection='3d')
# ax.plot_trisurf(np.array(x), np.array(y), np.array(circle_counts), linewidth=0, antialiased=True)
# plt.xlabel('param2')
# plt.ylabel('min_radius')
# plt.show()


def filter_circles(circles):
    circles = list(circles[0])

    strong_circles = []
    strong_circles.append(circles.pop())
    while circles:
        circle = circles.pop()
        suppress = False
        for strong_circle in strong_circles:
            # compute distance between circle centers and difference of radii relative to average circle radius
            # PARAMETERS: threshold distance
            # Hausdorff distance formula : https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisudagspnwAhVP7XMBHdIjDDAQFjABegQIAhAD&url=https%3A%2F%2Fhrcak.srce.hr%2Ffile%2F292566&usg=AOvVaw1hDMP5Es6tyxx0KICD7BfG
            total_radius = circle[2] + strong_circle[2]
            circle_coords = np.array([circle[0], circle[1]])
            strong_circle_coords = np.array([strong_circle[0], strong_circle[1]])
            distance = np.linalg.norm(circle_coords - strong_circle_coords)
            radius_difference = abs(int(strong_circle[2]) - int(circle[2]))
            hausdorff = distance + radius_difference
            if hausdorff / total_radius < 0.1:
                suppress = True
        if not suppress:
            strong_circles.append(circle)
    return np.array([strong_circles])


def circles_close_enough(circle1, circle2):
    total_radius = circle1[2] + circle2[2]
    circle1_coords = np.array([circle1[0], circle1[1]])
    circle2_coords = np.array([circle2[0], circle2[1]])
    distance = np.linalg.norm(circle1_coords - circle2_coords)
    radius_difference = abs(int(circle2[2]) - int(circle1[2]))
    hausdorff = distance + radius_difference
    return hausdorff / total_radius < 0.1


def circles_IOU_close_enough(circle1, circle2, thresh=0.8):
    r = circle1[2]
    R = circle2[2]
    circle1_coords = np.array([circle1[0], circle1[1]])
    circle2_coords = np.array([circle2[0], circle2[1]])
    d = np.linalg.norm(circle1_coords - circle2_coords)
    # intersection area formula: https: // mathworld.wolfram.com / Circle - CircleIntersection.html
    # https: // scipython.com / book / chapter - 8 - scipy / problems / p84 / overlapping - circles /
    if d < (r + R):
        minradius = min(r, R)
        maxradius = max(r, R)
        if d+minradius>maxradius:
            alpha = acos((r ** 2 + d ** 2 - R ** 2) / (2 * r * d))
            beta = acos((R ** 2 + d ** 2 - r ** 2) / (2 * R * d))
            int_area = alpha * r ** 2 + beta * R ** 2 - 0.5 * r ** 2 * sin(2 * alpha) - 0.5 * R ** 2 * sin(2 * beta)
            union_area = pi * r ** 2 + pi * R ** 2 - int_area
            IOU = int_area/union_area
        else:
            int_area = pi*minradius**2
            union_area = pi*maxradius**2
            IOU = int_area/union_area
    else:
        IOU = 0
    return IOU>thresh
