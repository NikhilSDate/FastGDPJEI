from experiments.tester import run_test, run_primitive_test, parse_annotations, run_time_test
from hyperopt import hp, fmin, tpe, Trials, space_eval, atpe
from hyperopt.fmin import generate_trials_to_calculate
from experiments.params import Params
from diagram_parser.diagram_graph_builder import parse_diagram, get_primitives_and_points, get_primitives
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import time


class ParamOptimizer:
    def __init__(self, objective, annotations_path, image_path, validation_split=0.3, seed=42):
        self.objective = objective
        self.annotations_path = annotations_path
        self.image_path = image_path
        random.seed(seed)
        files = list()
        for file_name, _, _, _ in parse_annotations(self.annotations_path):
            files.append(file_name)

        self.training = random.sample(files, int((1 - validation_split) * len(files)))
        self.validation = set(files).difference(self.training)

    def run_trial(self, input_path, output_path, space, max_evals):

        if input_path is None:
            trials = self.get_initial_trials(space)
        else:
            with open(input_path, 'rb') as f:
                trials = pickle.load(f)

        objective = lambda args: self.objective(args, self.training)
        best = fmin(objective, space, algo=atpe.suggest, max_evals=max_evals, trials=trials)
        with open(output_path, 'wb+') as f:
            pickle.dump(trials, f)

    def get_initial_trials(self, space):

        init_values = {}
        for key, _ in space.items():
            init_values[key] = Params.params[key]
        return generate_trials_to_calculate([init_values])

    def get_prititive_optimization_space(self):
        space = dict()
        space['min_radius_factor'] = hp.uniform('min_radius_factor', 0.05, 0.4)
        space['resize_dim'] = hp.quniform('resize_dim', 150, 350, 1)
        space['hough_circles_param_1'] = hp.uniform('hough_circles_param_1', 25, 100)
        space['circle_detector_is_a_circle_threshold'] = hp.quniform('circle_detector_is_a_circle_threshold',
                                                                     25, 99, 1)
        space['circle_detector_clustering_epsilon'] = hp.uniform('circle_detector_clustering_epsilon', 0.05, 0.2)
        space['line_detector_close_enough_angle_threshold'] = hp.uniform('line_detector_close_enough_angle_threshold',
                                                                         0.025,
                                                                         0.25)
        space['line_detector_close_enough_rho_threshold'] = hp.uniform('line_detector_close_enough_rho_threshold', 0.05,
                                                                       0.2)

        return space

    def get_point_optimization_space(self):
        space = dict()
        space['primitive_group_weight_offset_factor'] = hp.uniform('primitive_group_weight_offset_factor', 0.05, 0.5)
        space['text_detector_is_text_blob_low_thresh'] = hp.uniform('text_detector_is_text_blob_low_thresh', 10, 100)
        space['text_detector_is_text_blob_high_thresh'] = hp.uniform('text_detector_is_text_blob_high_thresh', 0.005,
                                                                     0.1)
        space['diagram_graph_builder_dbscan_eps'] = hp.uniform('diagram_graph_builder_dbscan_eps', 0.01, 0.2)
        space['corner_response_map_ksize'] = hp.quniform('corner_response_map_ksize', 1, 15, 1)
        space['corner_response_map_iters'] = hp.quniform('corner_response_map_iters', 1, 10, 1)
        space['character_detector_confusion_threshold'] = hp.quniform('character_detector_confusion_threshold', 10, 100,
                                                                      1)
        return space

    def point_optimization_objective(self, args, image_set):
        Params.update_params(args)
        file_scores, total_precision, total_recall = run_test(self.image_path, self.annotations_path, image_set)
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
        return -total_f1

    def primitive_optimization_objective(self, args, image_set):
        Params.update_params(args)
        total_precision, total_recall = run_primitive_test(self.image_path, self.annotations_path, image_set)
        f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
        return 1 - f1

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


optimizer = ParamOptimizer(ParamOptimizer.point_optimization_objective, 'data/images/annotations.xml', 'data/images')
file_set = set()
for file_name, _, _, _ in parse_annotations(optimizer.annotations_path):
    if len(file_name) == 7:
        file_set.add(file_name)


task = parse_diagram
run_time_test(task, 'data/images', 5, file_set)