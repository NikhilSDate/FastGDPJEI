import cv2.cv2 as cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def preprocess(image):
    blur = cv2.bilateralFilter(image, 15, 75, 75)

    return blur


def get_connected_components(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.bitwise_not(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1])  # ensure binary
    num_labels, labeled_image = cv2.connectedComponents(threshold)
    return labeled_image

def get_connected_component_points(labeled_image):
    points_dict = dict()
    for idx, label in np.ndenumerate(labeled_image):
        if not (label == 0):
            if label in points_dict:
                points_dict[label].append(idx)
            else:
                points_dict[label] = [idx]
    return points_dict


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


img = cv2.imread('../aaai/050.png')
img=preprocess(img)
labels = get_connected_components(img)
np_points=np.array(get_connected_component_points(labels)[6])
print(np_points)
y, x, h, w = cv2.boundingRect(np_points)
img_copy=img.copy()
cv2.rectangle(img_copy, (x-1, y-1), (x+w+1, y+h+1), (0,255,0), 1)
roi=img[y-1:y+h+1, x-1:x+w+1]
thresholded_roi=cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
data = pytesseract.image_to_data(roi, lang='eng', config='--psm 10')
print(data)
cv2.imshow('img',img_copy)
cv2.waitKey()
