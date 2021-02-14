# Python programe to illustrate
# corner detection with
# Harris Corner Detection Method

# organizing imports
import cv2.cv2 as cv2
import numpy as np
from utils.tools import freedman_diaconis_bins, otsus_threshold
import matplotlib.pyplot as plt
from diagram_parser.point_detector import remove_text

def distance(x1, y1, x2, y2):
    return np.linalg.norm((x2-x1, y2-y1))

# path to input image specified and
# image is loaded with imread command
image = cv2.imread('../aaai/054.png')
# convert the input image into
# grayscale color space
operatedImage = remove_text(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# modify the data type
# setting to 32-bit floating point
operatedImage = np.float32(operatedImage)

# apply the cv2.cornerHarris method
# to detect the corners with appropriate
# values as input parameters
dest = cv2.cornerHarris(operatedImage, 2, 3, 0.04)

# Results are marked through the dilated corners
dest = cv2.dilate(dest, None)

# Reverting back to the original image,
# with optimal threshold value
image[dest > 0.04 * dest.max()] = [0, 0, 255]
rows, cols= np.where(dest > 0.1 * dest.max())

# the window showing output image with corners
cv2.imshow('Image with Borders', image)

print(rows)
distances=[]
x, y = cols[1], rows[1]
for i in range(1, len(rows)):
    distances.append(distance(x, y, cols[i], rows[i]))
bins=freedman_diaconis_bins(distances)
hist = plt.hist(distances, bins)
print(hist)
thresh=otsus_threshold(hist)
print('thresh', thresh)
plt.show()
# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()




