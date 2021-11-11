import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import _csv as csv


def compare():
    with open('point_detection/point_complex_fastgdp.pickle', 'rb') as f:
        fastgdp_scores = [score[3] for score in pickle.load(f).values()]

    with open('point_detection/point_complex_geos.pickle', 'rb') as f:
        geos_scores = [score[3] for score in pickle.load(f).values()]
    print(np.mean(fastgdp_scores))
    print(np.mean(geos_scores))
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(fastgdp_scores, kde=True, ax=axs[0], bins=15)
    sns.histplot(geos_scores, kde=True, ax=axs[1], bins=15)
    plt.show()


def compute_metric(data, metric):
    if metric == 'macro_f1':
        f1_scores = [score[3] for score in data]
        return np.mean(f1_scores)
    elif metric == 'macro_f1_var':
        f1_scores = [score[3] for score in data]
        return np.var(f1_scores)
    elif metric == 'macro_precision':
        precisions = []
        for score in data:
            try:
                precisions.append(score[0] / score[1])
            except ZeroDivisionError:
                precisions.append(0)
        return np.mean(precisions)
    elif metric == 'macro_precision_var':
        precisions = []
        for score in data:
            try:
                precisions.append(score[0]/score[1])
            except ZeroDivisionError:
                precisions.append(0)
        return np.var(precisions)
    elif metric == 'macro_recall':
        recalls = []
        for score in data:
            try:
                recalls.append(score[0] / score[2])
            except ZeroDivisionError:
                recalls.append(0)
        return np.mean(recalls)
    elif metric == 'macro_recall_var':
        recalls = []
        for score in data:
            try:
                recalls.append(score[0] / score[2])
            except ZeroDivisionError:
                recalls.append(0)
        return np.var(recalls)
    elif metric == 'micro_f1':
        total_relevant = np.sum([score[0] for score in data])
        total_pred = np.sum([score[1] for score in data])
        total_gt = np.sum([score[2] for score in data])
        p = total_relevant / total_pred
        r = total_relevant / total_gt
        return 2 * p * r / (p + r)
    elif metric == 'micro_precision':
        total_relevant = np.sum([score[0] for score in data])
        total_pred = np.sum([score[1] for score in data])
        return total_relevant / total_pred
    elif metric == 'micro_recall':
        total_relevant = np.sum([score[0] for score in data])
        total_gt = np.sum([score[2] for score in data])
        return total_relevant / total_gt


def generate_csv(data1, data2, filename):
    header = ['metric', 'FastGDP', 'geosolver']
    metrics = ['macro_f1', 'macro_f1_var', 'macro_precision', 'macro_precision_var', 'macro_recall', 'macro_recall_var', 'micro_f1', 'micro_precision', 'micro_recall']
    with open(f'csv/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for metric in metrics:
            val1 = compute_metric(data1, metric)
            val2 = compute_metric(data2, metric)
            writer.writerow([metric, val1, val2])

with open('point_detection/point_complex_fastgdp.pickle', 'rb') as f:
    data1 = pickle.load(f)
with open('point_detection/point_complex_geos.pickle', 'rb') as f:
    data2 = pickle.load(f)
generate_csv(data1.values(), data2.values(), 'complex_point.csv')

# with open('time/test_fastgdp_complete_nolabel.pickle', 'rb') as f:
#     times1 = pickle.load(f)
# with open('time/test_geos_point_nolabel.pickle', 'rb') as f:
#     times2 = pickle.load(f)
# print(sum(times1.values()))
# print(sum(times2.values()))
# fig, axs = plt.subplots(ncols=2)
# sns.histplot(times1, kde=True, ax=axs[0], bins=15)
# sns.histplot(times2, kde=True, ax=axs[1], bins=15)
# plt.show()
