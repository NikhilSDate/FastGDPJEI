import numpy as np


class Params:
    params = {"hough_circles_param_1": 50, "hough_circles_param_2_range": (1, 100),
              "circle_detector_is_a_circle_threshold": 70,
              "circle_detector_clustering_epsilon": 0.1,
              "min_radius_factor": 0.2,
              "line_detector_close_enough_angle_threshold": 0.1,
              "line_detector_close_enough_rho_threshold": 0.075,
              "line_detector_clustering_eps": 0.025,
              "line_detector_canny_params": (50, 150, 3),
              "line_detector_hough_p_params": (1, np.pi / 180, 45, 10, 10),
              "line_detector_mode": 'hough_p_hesse',
              "text_detector_box_overlap_threshold": 0.7,
              "text_detector_is_text_blob_low_thresh": 25,
              "text_detector_is_text_blob_high_thresh": 0.1,
              "corner_detector_gaussian_blur_params": (5, 2),
              "corner_harris_params": (2, 3, 0.04),
              "corner_detector_is_corner_threshold": 0.04,
              "diagram_graph_builder_dbscan_eps": 0.05,
              "primitive_group_weight_offset_factor": 0.2,
              'resize_image_if_too_big': True,
              'resize_dim': 250,
              'diagram_parser_corner_lies_on_line_eps': 0.03,
              'circle_tangent_eps': 0.1,
               # second argument is confusion threshold
              'character_detector_mode': 'smart',
              'character_detector_confusion_threshold':50,
              # first elem is ksize, second argument is iterations
              'corner_response_map_ksize': 3,
              'corner_response_map_iters':2


              }

    @classmethod
    def update_params(cls, new_params):
        cls.params.update(new_params)
