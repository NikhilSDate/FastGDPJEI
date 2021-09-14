from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.utils import plot_model

from sklearn import metrics
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from nn.character_predictor import CharacterPredictor
train_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, width_shift_range=0.15, height_shift_range=0.15, rotation_range=15, brightness_range=[0.7, 1.3], zoom_range = [1, 1.2])
train_data = train_datagen.flow_from_directory('../English/Fnt/', subset='training', target_size=(14, 14),
                                            class_mode='categorical', batch_size=128, color_mode='grayscale', seed=42)
test_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

validation_data = test_datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                 class_mode='categorical', batch_size=128, color_mode='grayscale', seed=42, shuffle=False)
# def build_model():
#     model = models.Sequential()
#     model.add(Resizing(36, 36, input_shape=(14, 14, 1)))
#     model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
#     model.add(layers.MaxPooling2D())
#     model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D())
#     model.add(layers.Conv2D(filters=112, kernel_size=(3, 3), activation='relu'))
#     # model.add(layers.MaxPooling2D())
#     model.add(layers.Flatten())
#     model.add(layers.Dropout(rate=0.2))
#     model.add(layers.Dense(112, activation='relu'))
#     model.add(layers.Dropout(rate=0.2))
#     model.add(layers.Dense(62, activation='softmax'))
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#     return model

def get_confusion_matrix(model):
    Y_pred = model.predict(validation_data)
    y_pred = np.argmax(Y_pred, axis=1)
    confusion_matrix = metrics.confusion_matrix(validation_data.classes, y_pred)
    confused_indices = np.transpose(np.where(confusion_matrix>10))
    confused_indices_array = np.array([confused_indices[i] for i in range (len(confused_indices)) if confused_indices[i][0]!=confused_indices[i][1]])
    print(confused_indices_array)
    labels = CharacterPredictor().labels
    print(confused_indices_array[0])
    map_function = lambda x: (labels[int(x[0])], labels[int(x[1])])

    print([map_function(confused_index) for confused_index in confused_indices_array])
    sn.set(font_scale=0.5)
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
character_model = models.load_model(filepath='models/bayes_optimized_character_model.h5')
plot_model(character_model, show_shapes=True, show_dtype=True)