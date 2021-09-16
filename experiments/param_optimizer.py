from experiments.tester import run_test, run_primitive_test
from hyperopt import hp, fmin, tpe, Trials, space_eval
from experiments.params import Params
import pickle
import tensorflow as tf
import numpy as np


class ParamOptimizer:
    def __init__(self, objective):
        self.objective = objective

    def run_trial(self, input_path, output_path, space, max_evals):
        if input_path is None:
            trials = Trials()
        else:
            with open(input_path, 'rb') as f:
                trials = pickle.load(f)
        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        with open(output_path, 'wb+') as f:
            pickle.dump(trials, f)

    def get_prititive_optimization_space(self):
        space = dict()
        space['hough_circles_param_1'] = hp.choice('hough_circles_param_1', range(25, 100))
        space['circle_detector_objective_function_param_2_term_shape'] = hp.uniform(
            'circle_detector_objective_function_param_2_term_shape', -1, 5)

        space['circle_detector_objective_function_min_radius_term_shape'] = (
            hp.uniform('min_radius_term_0', 0, 2), hp.uniform('min_radius_term_1', 0, 1))
        space['circle_detector_min_radius_weight'] = hp.uniform('circle_detector_min_radius_weight', 0.1, 2)
        space['circle_detector_is_a_circle_threshold'] = hp.choice('circle_detector_is_a_circle_threshold',
                                                                   range(25, 99))
        space['line_detector_close_enough_angle_threshold'] = hp.uniform('line_detector_close_enough_angle_threshold',
                                                                         0.02,
                                                                         0.3)
        space['line_detector_close_enough_rho_threshold'] = hp.uniform('line_detector_close_enough_rho_threshold', 0.01, 0.1)
        return space
    def get_point_optimization_space(self):
        space = dict()
        space['primitive_group_weight_offset_factor'] = hp.uniform('primitive_group_weight_offset_factor', 0.05, 0.5)
        # space['text_detector_box_overlap_threshold'] = hp.uniform('text_detector_box_overlap_threshold', 0.5, 0.9)
        space['text_detector_is_text_blob_low_thresh'] = hp.uniform('text_detector_is_text_blob_low_thresh', 10, 100)
        space['text_detector_is_text_blob_high_thresh'] = hp.uniform('text_detector_is_text_blob_high_thresh', 0.005,
                                                                     0.1)
        space['diagram_graph_builder_dbscan_eps'] = hp.uniform('diagram_graph_builder_dbscan_eps', 0.01, 0.2)
        return space
    @staticmethod
    def point_optimization_objective(args):
        Params.update_params(args)
        f1_scores, total_precision, total_recall = run_test('../aaai', '../aaai/annotations_2.xml')
        print(total_precision, total_recall)
        f1_scores = np.array(f1_scores)
        loss = np.sum((1 - f1_scores) ** 2)

        return loss
    @staticmethod
    def primitive_optimization_objective(args):
        Params.update_params(args)
        total_precision, total_recall = run_primitive_test('../aaai', '../aaai/annotations_2.xml')
        f1 = 2*total_precision*total_recall/(total_precision+total_recall)
        return 1-f1
    @staticmethod
    def get_best_params(fp, space):
        with open(fp, 'rb+') as f:
            trials = pickle.load(f)
        best_params = dict()
        best = trials.best_trial['misc']['vals']
        for key, value in best.items():
            # remove array
            best_params[key] = value[0]
        return space_eval(space, best_params)
tf.get_logger().setLevel('ERROR')

# baseline = run_primitive_test('../validation/images', '../validation/annotations.xml')
# print(baseline)
# optimizer = ParamOptimizer(ParamOptimizer.primitive_optimization_objective)
# space = optimizer.get_prititive_optimization_space()
# best_params = ParamOptimizer.get_best_params('C:\\Users\cat\\PycharmProjects\\EuclideanGeometrySolver\\experiments\\optimization_results\\primitive_detection\\part_1.pickle', space)
# Params.update_params(best_params)
# optimized = run_primitive_test('../validation/images', '../validation/annotations.xml')
# print(optimized)

print(ParamOptimizer.point_optimization_objective({}))
#optimizer = ParamOptimizer(ParamOptimizer.primitive_optimization_objective)
# optimizer.run_trial('C:\\Users\cat\\PycharmProjects\\EuclideanGeometrySolver\\experiments\\optimization_results\\primitive_detection\\part_1.pickle',
#                    'C:\\Users\cat\\PycharmProjects\\EuclideanGeometrySolver\\experiments\\optimization_results\\primitive_detection\\part_1.pickle', optimizer.get_prititive_optimization_space(), 5)

