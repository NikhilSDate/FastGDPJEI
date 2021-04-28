import cv2.cv2 as cv2
import numpy as np
from diagram_parser.text_detector import remove_text
import matplotlib.pyplot as plt
import math
cv2.waitKey()
img = cv2.imread('NCERT-Solutions-for-Class-9-Maths-Chapter-10-Circles-Ex-10.4-Q1.png')
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
for param2 in range(15, 100):
    for min_radius in range(5, 100, 5):
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

def objective_function(param2, min_radius, num_circles, max_num_circles, max_param_2):
    if num_circles == 0:
        return 0
    else:
        return 1-(num_circles/max_num_circles)**3 +(param2/max_param_2)
comparator = lambda trial: objective_function(*trial, max_circles, 100)
sorted_results = sorted(trial_results, key=comparator)

best_params = sorted_results[-1]
print(sorted_results[-5:])
best_circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT,
                                1.5, 1, param1=50, param2=best_params[0],
                                minRadius=60, maxRadius=int(eroded.shape[0]/2))
best_circles = best_circles.astype(np.uint8)
for i in best_circles[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_trisurf(np.array(x), np.array(y), np.array(circle_counts), linewidth=0, antialiased=True)
plt.xlabel('param2')
plt.ylabel('min_radius')
plt.show()

def filter_circles(circles):
    circles = list(circles)

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
            if hausdorff/total_radius < 0.5:
                suppress = True
        if not suppress:
            strong_circles.append(circle)
    return np.array(strong_circles)


# = filter_circles(list(circles[0])).astype(np.int)
# Draw the circles

