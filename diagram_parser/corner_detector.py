# Python programe to illustrate
# corner detection with
# Harris Corner Detection Method

# organizing imports
import cv2.cv2 as cv2
import numpy as np
from diagram_parser.text_detector import remove_text
from diagram_parser.text_detector import connected_components_and_threshold
from experiments.params import Params


def distance(x1, y1, x2, y2):
    return np.linalg.norm((x2 - x1, y2 - y1))


def get_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked = remove_text(gray)
    # PARAM corner_detector_gaussian_blur_ksize
    # PARAM corner_detector_gaussian_blur_sigma
    blur_params = Params.params['corner_detector_gaussian_blur_params']
    masked = cv2.GaussianBlur(masked, (blur_params[0], blur_params[0]), blur_params[1])
    # PARAM harris_sobel_ksize
    # PARAM harris_blocksize
    # PARAM harris_k
    harris_params = Params.params['corner_harris_params']
    corners = cv2.cornerHarris(masked, harris_params[0], harris_params[1], harris_params[2])
    dilated_corners = cv2.dilate(corners, None)
    twice_dilated = cv2.dilate(dilated_corners, None)
    scaled = twice_dilated / twice_dilated.max()
    scaled *= 255
    scaled = np.maximum(np.zeros_like(scaled), scaled)
    int_image = np.zeros_like(scaled, dtype=np.uint8)
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            int_image[i][j] = round(scaled[i][j])
    _, int_image = cv2.threshold(int_image, 1, 255, cv2.THRESH_BINARY)

    filtered_dest = np.zeros_like(dilated_corners)
    # PARAM corner_detector_corner_threshold
    is_corner_thresh = Params.params['corner_detector_is_corner_threshold']
    filtered_dest[dilated_corners > is_corner_thresh * dilated_corners.max()] = 255
    uint8_filtered_dest = np.uint8(filtered_dest)
    (_, components, _, centroids), _, _ = connected_components_and_threshold(uint8_filtered_dest)

    return int_image, centroids[1:]


def draw_corners(image, points):
    image_copy = image.copy()
    for point in points:
        int_point = map(lambda f: int(f), point)  # convert coordinates to ints
        cv2.circle(image_copy, tuple(int_point), 2, [0, 255, 0], thickness=-1)
    return image_copy


# img = cv2.imread('../experiments/data/images/0030.png')
# corners = get_corners(img)
