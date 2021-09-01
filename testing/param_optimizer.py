from testing.tester import run_test
from hyperopt import hp, fmin, tpe
from testing.params import Params
import pickle
import tensorflow as tf
import numpy as np


def objective(args):
    Params.update_params(args)
    f1_scores, total_precision, total_recall = run_test()
    f1_scores = np.array(f1_scores)
    loss = np.sum((1-f1_scores)**2)
    # try:
    #     f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    # except ZeroDivisionError:
    #     return 0

    return loss


tf.get_logger().setLevel('ERROR')


def run_trial():
    space = dict()
    # space['hough_circles_param_1'] = hp.choice('hough_circles_param_1', range(25, 100))
    #space['circle_detector_objective_function_param_2_term_shape'] = hp.uniform(
    #    'circle_detector_objective_function_param_2_term_shape', -1, 5)

    space['circle_detector_objective_function_min_radius_term_shape'] = (
         hp.uniform('min_radius_term_0', 0, 2), hp.uniform('min_radius_term_1', 0, 1))
    space['circle_detector_min_radius_weight'] = hp.uniform('circle_detector_min_radius_weight', 0.1, 2)
    space['circle_detector_is_a_circle_threshold'] = hp.choice('circle_detector_is_a_circle_threshold', range(25, 99))
    space['line_detector_close_enough_angle_threshold'] = hp.uniform('line_detector_close_enough_angle_threshold', 0.02, 0.3)
    space['primitive_group_weight_offset_factor'] = hp.uniform('primitive_group_weight_offset_factor', 0.05, 0.5)
    space['line_detector_mode'] = hp.choice('line_detector_mode', ['hough_p_hesse', 'hough'])
    # space['text_detector_box_overlap_threshold'] = hp.uniform('text_detector_box_overlap_threshold', 0.5, 0.9)
    space['text_detector_is_text_blob_low_thresh'] = hp.uniform('text_detector_is_text_blob_low_thresh', 10, 100)
    space['text_detector_is_text_blob_high_thresh'] = hp.uniform('text_detector_is_text_blob_high_thresh', 0.005, 0.1)
    space['diagram_graph_builder_dbscan_eps'] = hp.uniform('diagram_graph_builder_dbscan_eps', 0.01, 0.2)
    baseline = objective({})
    print(f'baseline{baseline}')
    with open('optimization_part_5.pickle', 'rb') as f:
        trials = pickle.load(f)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=80, trials=trials)
    best['circle_detector_is_a_circle_threshold']+=25
    best['line_detector_mode'] = 'hough_p_hesse'
    with_optimization = objective(best)
    print(f'with_optimization{with_optimization}')
    # Params.update_params(best)
    # print(run_test())
    # with open('optimization_part_5.pickle', 'wb') as f:
    #     pickle.dump(trials, f)
run_trial()