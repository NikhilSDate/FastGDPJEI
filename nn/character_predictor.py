from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    def predict_character(self, image, character_mode='all'):
        shape = image.shape
        max_shape = max(shape[0], shape[1])
        width_padding = int((max_shape - shape[0]) / 2)
        height_padding = int((max_shape - shape[1]) / 2)
        bordered = cv2.copyMakeBorder(image, width_padding + 2, width_padding + 2, height_padding + 2,
                                      height_padding + 2,
                                      borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = np.reshape(bordered, newshape=(bordered.shape[0], bordered.shape[1], 1))
        img = np.expand_dims(img, axis=0)
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
            if predicted_index < 10:
                return character
            else:
                if (character, character.swapcase()) in self.confused_labels:
                    return character.upper()
                else:
                    return character
        elif character_mode == 'uppercase':
            predicted_index = np.argmax(y_pred[0, 10:36]) + 10
            character = self.labels[predicted_index]
            return character
        elif character_mode == 'lowercase':
            predicted_index = np.argmax(y_pred[0, 36:]) + 36
            character = self.labels[predicted_index]
            return character

    def is_upper(self, character_sequence):
        # TODO: CLEAN THIS UP
        is_letter = True
        for character_image in character_sequence:
            # shape = character_image.shape
            # max_shape = max(shape[0], shape[1])
            # width_padding = int((max_shape - shape[0]) / 2)
            # height_padding = int((max_shape - shape[1]) / 2)
            # bordered = cv2.copyMakeBorder(character_image, width_padding + 2, width_padding + 2, height_padding + 2,
            #                               height_padding + 2,
            #                               borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            # img = np.reshape(bordered, newshape=(bordered.shape[0], bordered.shape[1], 1))
            # img = np.expand_dims(img, axis=0)
            # y_pred = self.model.predict(img, batch_size=1)
            character = self.predict_character(character_image, character_mode='smart')

            if not character.isupper():
                is_letter = False
                break
        return is_letter

    def initialize_confused_labels(self):
        test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

        validation_data = test_datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                           class_mode='categorical', batch_size=128,
                                                           color_mode='grayscale', seed=42, shuffle=False)
        Y_pred = self.model.predict(validation_data)
        y_pred = np.argmax(Y_pred, axis=1)
        confusion_matrix = metrics.confusion_matrix(validation_data.classes, y_pred)
        # PARAM confusion threshold
        confused_indices = np.transpose(np.where(confusion_matrix > 50))
        confused_indices_array = np.array([confused_indices[i] for i in range(len(confused_indices)) if
                                           confused_indices[i][0] != confused_indices[i][1]])
        map_function = lambda x: (self.labels[int(x[0])], self.labels[int(x[1])])
        confused_labels = set([map_function(confused_index) for confused_index in confused_indices_array])
        return confused_labels
