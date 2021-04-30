import cv2.cv2 as cv2
import numpy as np
from diagram_parser.text_detector import remove_text
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN, MeanShift


def resize(image, final_larger_dim, interpolation=cv2.INTER_LINEAR):
    larger_dim = max(image.shape[0], image.shape[1])
    factor = final_larger_dim / larger_dim
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #    gray = resize(gray, 200)
    masked = remove_text(gray)
    # foreground is black so erode instead of dilating
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(masked, kernel, iterations=1)
    blur = cv2.GaussianBlur(eroded, (3, 3), 1.5)
    return blur


def get_best_params(image, objective_function):
    trial_results = []
    x = []
    y = []
    circle_counts = []
    max_circles = 0
    for param2 in range(1, 100, 2):
        for min_radius in range(0, int(image.shape[0] / 2), 2):
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, 1,
                                       param1=50, param2=param2, minRadius=min_radius,
                                       maxRadius=int(image.shape[0] / 2))
            if circles is not None:
                circles = circles.astype(np.uint8)
                num_circles = circles.shape[1]

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
    comparator = lambda trial: objective_function(*trial, max_circles, 100, int(image.shape[0] / 2))
    sorted_results = sorted(trial_results, key=comparator)
    params = sorted_results[-1]
    return params


def objective_function(param2, min_radius, num_circles, max_num_circles, max_param_2, max_radius):
    if num_circles == 0:
        return 0
    else:
        param2_term = 1 - math.exp(-5 * (param2 / max_param_2))
        min_radius_term = math.exp((-(min_radius / max_radius) ** 2))
        min_radius_term = 1 - (1.25 * (min_radius / max_radius - 0.2) ** 2)
        return param2_term + min_radius_term


def show_circles(image, circles):
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in circles[0, :]:
        cv2.circle(color, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(color, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', color)
    cv2.waitKey()
def clustering_filter(circles):
    circle_centers = [(circle[0], circle[1]) for circle in circles[0]]
    clustering = DBSCAN(eps=0.05 * (img.shape[0] + img.shape[1]), min_samples=2).fit(circle_centers)
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

    return np.array([filtered_circles])

img = cv2.imread('NCERT-Solutions-for-Class-9-Maths-Chapter-10-Circles-Ex-10.4-Q4.1.png')
img = preprocess(img)
best_params = get_best_params(img, objective_function=objective_function)
best_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                1.5, 1, param1=50, param2=best_params[0],
                                minRadius=best_params[1], maxRadius=int(img.shape[0] / 2))
best_circles = best_circles.astype(np.uint8)
filtered_circles = clustering_filter(best_circles)
print(best_params)
cv2.imshow('preprocessed', img)
show_circles(img, filtered_circles.astype(np.uint16))
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
