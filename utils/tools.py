from scipy import stats
from math import ceil
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
from math import floor


def freedman_diaconis_bins(data):
    iqr = stats.iqr(data)
    width = 2 * iqr / pow(len(data), 1 / 3)
    range = max(data) - min(data)
    num_bins = ceil(range / width)
    return num_bins


# data must be sorted
def otsus_threshold(data):
    running_total = 0
    best_threshold = [data[0], np.var(data)]
    datapoints = len(data)
    for i in range(1, datapoints):
        running_total = running_total + 1
        # TODO:fix infinite variance for last step
        weighted_var = (running_total * np.var(data[0:i]) + (datapoints - running_total) * np.var(data[i:]))

        if weighted_var < best_threshold[1]:
            best_threshold[0] = data[i]
            best_threshold[1] = weighted_var

    return best_threshold[0]


def var(data, weights):
    return DescrStatsW(data, weights=weights, ddof=0).var


def round_up_to_multiple(x, multiple):
    return ceil(x / multiple) * multiple


def round_down_to_multiple(x, multiple):
    return floor(x / multiple) * multiple
