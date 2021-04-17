import cv2.cv2 as cv2
import numpy as np

from utils.vector import draw_vector_contours
from utils.tools import freedman_diaconis_bins, otsus_threshold



def preprocess(image):
    blur = cv2.bilateralFilter(image, 9, 75, 75)

    return blur


def threshold_image(image):
    thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    return thresh


def connected_components_and_threshold(image):
    gray = image

    threshold = cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)[1]  # ensure binary

    components = cv2.connectedComponentsWithStats(threshold, connectivity=8)
    stats = components[2]
    small_areas = stats[1:, 4]
    area_threshold = otsus_threshold(sorted(small_areas))
    return components, area_threshold


def get_component_roi(image, stats, index):
    stat = stats[index]
    x = stat[0]
    y = stat[1]
    w = stat[2]
    h = stat[3]
    return image[y:y + h, x:x + w]


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


def white_out_components_with_threshold(image, components, threshold):
    labels = components[0]

    stats = components[1]
    rejected_labels = []
    mask = np.zeros_like(image)
    for idx, stat in enumerate(stats):
        if stat[4] < threshold:
            rejected_labels.append(idx)
    for idx, label in np.ndenumerate(labels):
        if label in rejected_labels:
            mask[idx] = 255

    filtered_image = cv2.add(image, mask)

    return filtered_image


def remove_text(image):
    (_, labels, stats, _), area_threshold = connected_components_and_threshold(cv2.bitwise_not(image))
    masked_image = white_out_components_with_threshold(image, (labels, stats), area_threshold)
    blur = cv2.bilateralFilter(masked_image, 5, 75, 75)
    return blur


def text_components_with_centroids(image):
    # TODO:CLEAN UP CODE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text_centroids = []
    text_areas = []
    text_regions = []
    (_, labels, stats, centroids), area_threshold = connected_components_and_threshold(cv2.bitwise_not(image))
    for idx, stat in enumerate(stats):
        if 20 < stat[4] < area_threshold:
            text_centroids.append(centroids[idx])
            text_areas.append(stat[4])
            text_regions.append(get_component_roi(image, stats, idx))
    return text_centroids, text_regions


def resize_region(region, ):

    bordered = cv2.copyMakeBorder(region, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    thresholded = cv2.threshold(bordered, -1, 255, cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresholded), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('region', cv2.bitwise_not(thresholded))
    cv2.waitKey()
    approximate_contours = [cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True) for contour in contours]
    holes = [approximate_contours[i] for i in range(len(contours)) if hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1]
    inverted = cv2.cvtColor(cv2.bitwise_not(bordered), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(inverted, contours, 1, thickness=1, color=[255, 0, 0])

img = cv2.imread('../aaai/035.png')
_, text_regions = text_components_with_centroids(img)
text_region = text_regions[1]
#resize_region(text_region)
bordered = cv2.copyMakeBorder(text_region, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

big_image = cv2.resize(bordered, (40,40), interpolation=cv2.INTER_CUBIC)
thresholded = cv2.threshold(big_image, -1, 255, cv2.THRESH_OTSU)[1]
cv2.imshow('big', big_image)
print(text_region.shape)
cv2.waitKey()

