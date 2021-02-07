import cv2.cv2 as cv2
import numpy as np

img = cv2.imread('../aaai/006.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur=cv2.bilateralFilter(gray,15,10,10)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 1,
                           param1=60, param2=40, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
# Draw the circles
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
