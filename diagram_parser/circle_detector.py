import cv2.cv2 as cv2
import numpy as np
from diagram_parser.text_detector import remove_text
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
cv2.waitKey()
img = cv2.imread('../aaai/055.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

masked=remove_text(gray)
# foreground is black so erode instead of dilating
kernel = np.ones((2,2), np.uint8)
eroded = cv2.erode(masked, kernel, iterations=1)
eroded = cv2.GaussianBlur(eroded, (3, 3), 1.5)
thresholded = cv2.threshold(eroded, 230, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('eroded', eroded)
trial_results = []
x=[]
y=[]
circle_counts = []
max_circles = 0
for param2 in range(15, 100, 2):
    for min_radius in range(5, 100, 2):
        circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT, 1.5, 1,
                           param1=50, param2=param2, minRadius=min_radius, maxRadius=int(masked.shape[0]/2))
        if circles is not None:
            circles= circles.astype(np.uint8)
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

def objective_function(param2, min_radius, num_circles, max_num_circles, max_param_2, max_radius):
    if num_circles == 0:
        return 0
    else:
        circle_count_term = 0.2*(1-(num_circles/max_num_circles)**3)
        param2_term = 1 - math.exp(-5*(param2/max_param_2))
        min_radius_term = math.exp((-(min_radius / max_radius) ** 2))
        return param2_term + min_radius_term
comparator = lambda trial: objective_function(*trial, max_circles, 100, int(masked.shape[0]/2))
sorted_results = sorted(trial_results, key=comparator)
params = sorted_results[-1]
print(objective_function(40, 5, 10, 100, 100, 100))
best_circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT,
                                1.5, 1, param1=50, param2=params[0],
                                minRadius=params[1], maxRadius=int(eroded.shape[0] / 2))
print(params)
print(best_circles)
circle_centers = [(circle[0], circle[1]) for circle in best_circles[0]]
clustering = DBSCAN(eps=10, min_samples=1).fit(circle_centers)
print(clustering.labels_)
best_circles = best_circles.astype(np.uint8)

def filter_circles(circles):
    circles = list(circles[0])

    strong_circles=[]
    strong_circles.append(circles.pop())
    while circles:
        circle = circles.pop()
        suppress = False
        for strong_circle in strong_circles:
            # compute distance between circle centers and difference of radii relative to average circle radius
            # PARAMETERS: threshold distance
            # Hausdorff distance formula : https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisudagspnwAhVP7XMBHdIjDDAQFjABegQIAhAD&url=https%3A%2F%2Fhrcak.srce.hr%2Ffile%2F292566&usg=AOvVaw1hDMP5Es6tyxx0KICD7BfG
            total_radius = circle[2]+strong_circle[2]
            circle_coords = np.array([circle[0], circle[1]])
            strong_circle_coords = np.array([strong_circle[0], strong_circle[1]])
            distance = np.linalg.norm(circle_coords-strong_circle_coords)
            radius_difference = abs(int(strong_circle[2])-int(circle[2]))
            hausdorff = distance + radius_difference
            if hausdorff/total_radius < 0.1:
                suppress = True
        if not suppress:
            strong_circles.append(circle)
    return np.array([strong_circles])


# ax = fig.gca(projection='3d')
# ax.plot_trisurf(np.array(x), np.array(y), np.array(circle_counts), linewidth=0, antialiased=True)
# plt.xlabel('param2')
# plt.ylabel('min_radius')
# plt.show()
for i in best_circles[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey(0)


