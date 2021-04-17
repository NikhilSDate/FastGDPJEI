from math import floor
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2.cv2 as cv2
class ImageGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, dim=(80, 80), n_channels=1,
             n_classes=62, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_paths = image_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __getitem__(self, index):
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = [self.labels[k] for k in batch_indices]
        batch_paths = [self.image_paths[k] for k in batch_indices]
        X, y = self.generate_images(batch_paths, batch_labels)
        return X, y
    def generate_data(self, paths, labels):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype = np.uint8)
        for idx, path in enumerate(paths):
            X[idx] = cv2.imread(path)

        X/=255
        y = to_categorical(labels, num_classes=self.n_classes)
        return X, y

    def __len__(self):
        return floor(len(self.list_IDs)/self.batch_size)
