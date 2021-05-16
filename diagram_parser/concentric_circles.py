import cv2.cv2 as cv2
import numpy as np
img = cv2.imread('test_images/concentric_circles.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1,
                           param1=50, param2=99, minRadius=20,
                           maxRadius=int((img.shape[0] + img.shape[1]) / 4))
circles = circles.astype(np.uint8)
for i in circles[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey()