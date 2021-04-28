from tensorflow.keras import models
import string
import cv2.cv2 as cv2
import numpy as np
class CharacterPredictor():

    def __init__(self, model_path='../nn/bayes_optimized_character_model.h5'):
        self.model = models.load_model(model_path)
        self.labels = dict()
        valid_characters = list(string.digits)+list(string.ascii_uppercase)+list(string.ascii_lowercase)
        self.labels = {i:valid_characters[i] for i in range(len(valid_characters))}

    def predict_character(self, image):
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
        predicted_index = np.argmax(y_pred)
        character = self.labels[predicted_index]
        return character

