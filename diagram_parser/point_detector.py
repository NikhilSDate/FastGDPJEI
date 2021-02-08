import cv2.cv2 as cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import freedman_diaconis_bins

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def preprocess(image):
    blur = cv2.bilateralFilter(image, 15, 75, 75)

    return blur


def get_connected_components(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = cv2.bitwise_not(cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)[1])  # ensure binary

    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)

    return labeled_image, stats

def get_component_roi(image, stats, index):
    stat=stats[index]
    x = stat[0]
    y = stat[1]
    w = stat[2]
    h = stat[3]
    return image[y:y+h, x:x+w]
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


img = cv2.imread('../aaai/032.png')
img=preprocess(img)
labels, stats = get_connected_components(img)
roi=get_component_roi(img, stats, 3)
areas=stats[1:, 4]
plt.hist(areas, bins=freedman_diaconis_bins(areas))
plt.show()
bordered_image=cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255,255,255))
resized_image=cv2.resize(bordered_image, (0, 0), fx=2, fy=2)
data = pytesseract.image_to_data(resized_image, lang='eng', config='--psm 10 -c page_separator=""')
print(data)

cv2.imshow('border', resized_image)
cv2.waitKey()
# cv2.imshow('img',img_copy)
# cv2.waitKey()
