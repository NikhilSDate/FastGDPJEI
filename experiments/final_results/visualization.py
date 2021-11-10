import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def compare():
    with open('point_detection/point_train_fastgdp.pickle', 'rb') as f:
        fastgdp_scores = [score[3] for score in pickle.load(f).values()]

    with open('point_detection/point_train_geos.pickle', 'rb') as f:
        geos_scores = [score[3] for score in pickle.load(f).values()]
    print(np.mean(geos_scores))
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(fastgdp_scores, kde=True, ax=axs[0], bins=15)
    sns.histplot(geos_scores, kde=True, ax=axs[1], bins=15)
    plt.show()

with open('time/complex_fastgdp_complete_nolabel.pickle', 'rb') as f:
    times1 = pickle.load(f)
with open('time/complex_geos_point_nolabel.pickle', 'rb') as f:
    times2 = pickle.load(f)
print(sum(times1.values()))
print(sum(times2.values()))
fig, axs = plt.subplots(ncols=2)
sns.histplot(times1, kde=True, ax=axs[0], bins=15)
sns.histplot(times2, kde=True, ax=axs[1], bins=15)
plt.show()