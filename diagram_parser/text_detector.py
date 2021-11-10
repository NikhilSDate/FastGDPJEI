import cv2.cv2 as cv2
import numpy as np
from numpy import cos, sin
from experiments.params import Params


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
    stats_with_small_areas = stats[1:]
    areas = [stat[2]*stat[3] for stat in stats_with_small_areas]
    if len(areas) < 2:
        return components, -1, threshold
    area_threshold = sorted(areas)[-2]

    return components, area_threshold, threshold


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
        if stat[2]*stat[3] <= threshold:
            rejected_labels.append(idx)
    for idx, label in np.ndenumerate(labels):
        if label in rejected_labels:
            mask[idx] = 255

    filtered_image = cv2.add(image, mask)

    return filtered_image


def remove_text(image):
    # TODO: fixed intermediate blurring

    # does not convert image to grayscale
    (_, labels, stats, _), area_threshold, _ = connected_components_and_threshold(cv2.bitwise_not(image))
    masked_image = white_out_components_with_threshold(image, (labels, stats), area_threshold)
    #    blur = cv2.bilateralFilter(masked_image, 5, 75, 75)
    return masked_image


def text_components_with_centroids(image):
    # TODO:FIX LOWER THRESHOLD
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text_regions = dict()
    (_, labels, stats, centroids), area_threshold, thresholded = connected_components_and_threshold(
        cv2.bitwise_not(image))
    bounding_rects = list()
    for idx, stat in enumerate(stats):
        # TODO: IMPLEMENT SMARTER METHOD FOR FIGURING OUT IF THE BLOB CONTAINS TEXT OR NOT
        low = Params.params['text_detector_is_text_blob_low_thresh']
        high = Params.params['text_detector_is_text_blob_high_thresh']
        if low < stat[2]*stat[3] <= (high * image.shape[0] * image.shape[1]):
            # cv2.imshow(str(idx), get_component_roi(image, stats, idx))
            # cv2.waitKey()
            points = []
            for y in range(stat[1], stat[1] + stat[3]):
                for x in range(stat[0], stat[0] + stat[2]):
                    if thresholded[y][x] == 255:
                        points.append((x, y))
            points = np.array(points, dtype=np.float32)
            rect = cv2.boundingRect(points)

            bounding_rects.append((idx, rect))
#            cv2.imshow(str(idx), get_component_roi(image, stats, idx))
#    cv2.waitKey()

    bounding_rects = sorted(bounding_rects, key=lambda rect: rect[1][0])
    used_rects = list()
    for i in range(len(bounding_rects)):
        used_rects.append(False)
    accepted_boxes = list()
    for i1, rect1 in enumerate(bounding_rects):
        if not used_rects[i1]:
            indices = [rect1[0]]
            currentx1 = rect1[1][0]
            currenty1 = rect1[1][1]
            currentx2 = currentx1 + rect1[1][2]
            currenty2 = currenty1 + rect1[1][3]
            used_rects[i1] = True
            for i2, rect2 in enumerate(bounding_rects[i1 + 1:], start=i1 + 1):
                nextx1 = rect2[1][0]
                nexty1 = rect2[1][1]
                nextx2 = nextx1 + rect2[1][2]
                nexty2 = nexty1 + rect2[1][3]

                boundx1 = min(currentx1, nextx1)
                boundx2 = max(currentx2, nextx2)
                boundy1 = min(currenty1, nexty1)
                boundy2 = max(currenty2, nexty2)
                bound_area = (boundx2 - boundx1) * (boundy2 - boundy1)
                current_box_area = (currentx2 - currentx1) * (currenty2 - currenty1)
                next_box_area = (nextx2 - nextx1) * (nexty2 - nexty1)
                # PARAM: TEXT_DETECTOR_MERGE_BOXES_THRESHOLD
                overlap_threshold = Params.params['text_detector_box_overlap_threshold']
                if ((current_box_area + next_box_area) / bound_area) >= overlap_threshold:
                    currentx1 = boundx1
                    currentx2 = boundy2
                    currenty1 = boundy1
                    currenty2 = boundy2
                    indices.append(rect2[0])
                    used_rects[i2] = True
            accepted_boxes.append((indices, (currentx1, currenty1, currentx2, currenty2)))
    for indices, box in accepted_boxes:
        weighted_sum = 0
        total_area = 0
        characters = list()
        for index in indices:
            weighted_sum += centroids[index] * stats[index][4]
            total_area += stats[index][4]
            characters.append(get_component_roi(image, stats, index))

        centroid = weighted_sum / total_area
        text_regions[tuple(centroid)] = characters
    return text_regions


def rotate_points(points, angle):
    rotation_matrix = np.array([[cos(angle), -sin(angle)],
                                [sin(angle, cos(angle))]])
    transposed_points = np.transpose(points)
    rotated_points = np.transpose(np.matmul(rotation_matrix, transposed_points))
    return rotated_points


def resize_region(region, ):
    bordered = cv2.copyMakeBorder(region, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    thresholded = cv2.threshold(bordered, -1, 255, cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresholded), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('region', cv2.bitwise_not(thresholded))
    cv2.waitKey()
    approximate_contours = [cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True) for contour in
                            contours]
    holes = [approximate_contours[i] for i in range(len(contours)) if
             hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1]
    inverted = cv2.cvtColor(cv2.bitwise_not(bordered), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(inverted, contours, 1, thickness=1, color=[255, 0, 0])
# image = cv2.imread('../symbols/100.png')
# components = text_components_with_centroids(image)
