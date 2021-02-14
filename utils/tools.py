from scipy import stats
from math import ceil
from statsmodels.stats.weightstats import DescrStatsW


def freedman_diaconis_bins(data):
    iqr = stats.iqr(data)
    width = 2 * iqr / pow(len(data), 1 / 3)
    range = max(data) - min(data)
    num_bins = ceil(range / width)
    return num_bins


def otsus_threshold(dist):
    weights = dist[0]
    bins = dist[1]
    running_total = 0
    best_threshold = [bins[0], var(bins[0:-1], weights)]
    print('best_threshold', best_threshold)
    num_bins = len(bins)
    datapoints = sum(weights)
    for i in range(1, num_bins):

        running_total = running_total + weights[i - 1]
        # TODO:fix infinite variance for last step
        weighted_var = (running_total * var(bins[0:i], weights[0:i]) + (datapoints - running_total) * var(bins[i:-1],
                                                                                                         weights[i:]))/datapoints
        print(bins[i], weighted_var)
        if weighted_var < best_threshold[1]:
            best_threshold[0] = bins[i]
            best_threshold[1] = weighted_var

    return best_threshold[0]


def var(data, weights):
    return DescrStatsW(data, weights=weights, ddof=0).var
