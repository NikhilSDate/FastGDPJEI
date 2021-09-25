from experiments.tester import run_test, run_primitive_test, parse_annotations
from hyperopt import hp, fmin, tpe, Trials, space_eval, atpe
from experiments.params import Params
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
        with open(output_path, 'wb+') as f:
            pickle.dump([], f)
        if input_path is None:
            trials = Trials()
        else:
            with open(input_path, 'rb') as f:
                trials = pickle.load(f)
        objective = lambda args: self.objective(args, self.training)
        best = fmin(objective, space, algo=atpe.suggest, max_evals=max_evals, trials=trials)
        with open(output_path, 'wb+') as f:
            pickle.dump(trials, f)

    def get_prititive_optimization_space(self):
        space = dict()
        space['circle_detector_objective_function_param_2_term_shape'] = hp.uniform(
            'circle_detector_objective_function_param_2_term_shape', -1, 5)

        space['circle_detector_objective_function_min_radius_term_shape'] = (
            hp.uniform('min_radius_term_0', 0, 2), hp.uniform('min_radius_term_1', 0, 1))
        space['circle_detector_min_radius_weight'] = hp.uniform('circle_detector_min_radius_weight', 0.1, 2)
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
        return space

    @staticmethod
    def point_optimization_objective(args, image_set):
        Params.update_params(args)
        f1_scores, total_precision, total_recall = run_test('data/images', 'data/annotations.xml', image_set)
        f1_scores = np.array(f1_scores)
        loss = np.sum((1 - f1_scores) ** 2)

        return loss

    @staticmethod
    def primitive_optimization_objective(args, image_set):
        Params.update_params(args)
        total_precision, total_recall = run_primitive_test('data/images', 'data/annotations.xml', image_set)
        f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
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


optimizer = ParamOptimizer(ParamOptimizer.primitive_optimization_objective, 'data/annotations.xml', 'data/images')
space = optimizer.get_prititive_optimization_space()
optimizer.run_trial(None, 'optimization_results/primitive_detection_with_stopping/part_1.pickle', space, 20)
# with open('optimization_results/primitive_detection_with_stopping/2.pickle', 'rb+') as f:
#     trials = pickle.load(f)
# validation_losses = []
# count = 0
# for trial in trials.trials:
#
#     raw_params = trial['misc']['vals']
#     params = dict()
#     for key, value in raw_params.items():
#         # remove array
#         params[key] = value[0]
#
#     params = space_eval(space, params)
#     before = round(time.time() * 1000)
#     validation_losses.append(ParamOptimizer.primitive_optimization_objective(params, optimizer.validation))
#     after = round(time.time() * 1000)
#     count += 1
#     if count == 25:
#         break
#     print(f'trials done {count}. {round((after-before)/1000)}s/trial')
# with open('optimization_results/primitive_detection_with_stopping/val_results', 'wb+') as f:
#     pickle.dump(validation_losses, f)
