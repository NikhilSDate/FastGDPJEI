from scipy import stats
from math import ceil
def freedman_diaconis_bins(data):
    iqr=stats.iqr(data)
    width=2*iqr/pow(len(data), 1/3)
    range=max(data)-min(data)
    num_bins= ceil(range/width)
    return num_bins
