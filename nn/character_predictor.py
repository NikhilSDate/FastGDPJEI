from tensorflow.keras import models
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from experiments.params import Params
import string
import cv2.cv2 as cv2

import numpy as np
from sklearn import metrics


class CharacterPredictor:
    confused_labels = None

    def __init__(self, model_path='../nn/models/bayes_optimized_character_model.h5'):

        self.model = models.load_model(model_path)

        self.labels = dict()
        valid_characters = list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)
        self.labels = {i: valid_characters[i] for i in range(len(valid_characters))}
        if CharacterPredictor.confused_labels is None:
            CharacterPredictor.confused_labels = self.initialize_confused_labels()

    def preprocess_image(self, image):
        # TODO: pad after resizing
        shape = image.shape
        y = shape[0]
        x = shape[1]
        if x < y:
            top = 0
            bottom = 0
            diff = y - x
            if diff % 2 == 0:
                left = int(diff / 2)
                right = int(diff / 2)
            else:
                left = int(diff / 2)
                right = int(diff / 2) + 1
        elif x > y:
            left = 0
            right = 0
            diff = x - y
            if diff % 2 == 0:
                top = int(diff / 2)
                bottom = int(diff / 2)
            else:
                top = int(diff / 2)
                bottom = int(diff / 2) + 1
        else:
            top = 0
            bottom = 0
            left = 0
            right = 0

        bordered = cv2.copyMakeBorder(image, top=top + 2, bottom=bottom + 2, left=left + 2, right=right + 2,
                                      borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "EDSR_x3.pb"
        # sr.readModel(path)
        #
        # # Set the desired model and scale to get correct pre- and post-processing
        # sr.setModel("edsr", 3)
        #
        # # Upscale the image
        # result = sr.upsample(image)
        # cv2.imshow('resized image', result)
        bordered = cv2.resize(bordered, (14, 14), interpolation=cv2.INTER_LINEAR)

        img = np.reshape(bordered, newshape=(bordered.shape[0], bordered.shape[1], 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict_character(self, image, character_mode='all'):
        img = self.preprocess_image(image)
        y_pred = self.model.predict(img, batch_size=1)
        if character_mode == 'all':
            predicted_index = np.argmax(y_pred)
            character = self.labels[predicted_index]
            return character
        elif character_mode == 'numbers':
            predicted_index = np.argmax(y_pred[0, 0:10])
            character = self.labels[predicted_index]
            return character
        elif character_mode == 'letters':
            predicted_index = np.argmax(y_pred[0, 10:]) + 10
            character = self.labels[predicted_index]
            return character
        elif character_mode == 'smart':
            predicted_index = np.argmax(y_pred)
            character = self.labels[predicted_index]
            if (character, character.swapcase()) in self.confused_labels:
                return character.upper()
            else:
                for letter in string.ascii_letters:
                    if (character, letter) in self.confused_labels:
                        return letter.upper()

                return character

        elif character_mode == 'uppercase':
            predicted_index = np.argmax(y_pred[0, 10:36]) + 10
            character = self.labels[predicted_index]
            return character
        elif character_mode == 'lowercase':
            predicted_index = np.argmax(y_pred[0, 36:]) + 36
            character = self.labels[predicted_index]
            return character

    def upper_confidence(self, image):
        img = self.preprocess_image(image)
        y_pred = self.model.predict(img, batch_size=1)
        return np.sum(y_pred[0, 10:36])

    def is_upper(self, character_sequence):
        is_upper = True
        for character_image in character_sequence:
            mode = Params.params['character_detector_mode']
            character = self.predict_character(character_image, character_mode=mode)

            if not character.isupper():
                is_upper = False
                break
        return is_upper, self.upper_confidence(character_sequence[0])

    def initialize_confused_labels(self):
        test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

        validation_data = test_datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                           class_mode='categorical', batch_size=128,
                                                           color_mode='grayscale', seed=42, shuffle=False)
        Y_pred = self.model.predict(validation_data)
        y_pred = np.argmax(Y_pred, axis=1)
        confusion_matrix = metrics.confusion_matrix(validation_data.classes, y_pred)
        # PARAM confusion threshold
        # confusion_threshold = Params.params['character_detector_confusion_threshold']
        confusion_threshold = Params.params['character_detector_confusion_threshold']
        confused_indices = np.transpose(np.where(confusion_matrix > confusion_threshold))
        confused_indices_array = np.array([confused_indices[i] for i in range(len(confused_indices)) if
                                           confused_indices[i][0] != confused_indices[i][1]])
        map_function = lambda x: (self.labels[int(x[0])], self.labels[int(x[1])])

        confused_labels = set([map_function(confused_index) for confused_index in confused_indices_array])
        return confused_labels
