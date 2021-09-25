import pickle
import numpy as np
import matplotlib.pyplot as plt
with open('optimization_results/primitive_detection_with_stopping/val_results', 'rb+') as f:
    validation_losses = pickle.load(f)
with open('optimization_results/primitive_detection_with_stopping/2.pickle', 'rb+') as f:
    training_losses = pickle.load(f).losses()
x1 = list(range(60))
x2 = list(range(25))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x2, validation_losses)
ax.plot(x1, training_losses)
plt.show()

with open('optimization_results/primitive_detection/part_3.pickle', 'rb+') as f:
    trials1 = pickle.load(f)
    print(trials1.best_trial['misc']['vals'])

with open('optimization_results/primitive_detection_with_stopping/2.pickle', 'rb+') as f:
    trials2 = pickle.load(f)
    print(trials2.best_trial['misc']['vals'])

with open('optimization_results/point_detection/optimization_part_5.pickle', 'rb+') as f:
    trials3 = pickle.load(f)
    print(trials3.best_trial['misc']['vals'])


