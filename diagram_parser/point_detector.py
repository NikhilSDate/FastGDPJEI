import cv2.cv2 as cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import freedman_diaconis_bins, otsus_threshold
from diagram_parser.line_detecter import LineDetector

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def preprocess(image):
    blur = cv2.bilateralFilter(image, 9, 75, 75)

    return blur


def threshold_image(image):
    thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    return thresh


def get_connected_components(image):
    gray = image

    threshold = cv2.bitwise_not(cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)[1])  # ensure binary

    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)

    return labeled_image, stats


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
    print(rejected_labels)
    for idx, label in np.ndenumerate(labels):
        if label in rejected_labels:
            mask[idx] = 255

    print('shape', image.shape, mask.shape)
    filtered_image = cv2.add(image, mask)

    return filtered_image

def remove_text(image):
    preprocessed = preprocess(image)
    labels, stats = get_connected_components(image)
    areas = stats[1:, 4]
    hist = np.histogram(areas, bins=freedman_diaconis_bins(areas))
    threshold = otsus_threshold(hist)

    masked_image = white_out_components_with_threshold(image, (labels, stats), threshold)
    blur=cv2.bilateralFilter(masked_image, 5, 75, 75)
    return blur

img = cv2.imread('../aaai/058.png', 0)
masked=remove_text(img)


# cv2.imshow('image', masked)
# cv2.waitKey()
# detector=LineDetector(cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR))
# lines=detector.get_filtered_lines()
# cv2.imshow('border', detector.draw_lines(lines))
# cv2.waitKey()



# bordered_image = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
# resized_image = cv2.resize(bordered_image, (0, 0), fx=2, fy=2)
# data = pytesseract.image_to_data(resized_image, lang='eng', config='--psm 10 -c page_separator=""')
# print(data)
# cv2.imshow('img',img_copy)
# cv2.waitKey()
