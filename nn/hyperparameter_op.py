from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def build_model_hyperband(hp):
    model = models.Sequential()
    image_size = hp.Int('image_size', 26, 46, step=10)
    model.add(Resizing(image_size, image_size))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())
    last_conv_filters = hp.Int('last_conv_filters', 64, 128, step=16)
    model.add(layers.Conv2D(filters=last_conv_filters, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.4))
    dense_units = hp.Int('dense_units', 64, 128, step=16)
    model.add(layers.Dense(dense_units, activation='relu'))

    model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(62, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    return model

def build_model_bayes(hp):
    model = models.Sequential()
    model.add(Resizing(46, 46))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())
    last_conv_filters = hp.Int('last_conv_filters', 64, 128, step=16)
    model.add(layers.Conv2D(filters=last_conv_filters, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    dropout_rate = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)
    model.add(layers.Dropout(rate=dropout_rate))
    dense_units = hp.Int('dense_units', 64, 128, step=16)
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(62, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    return model
def optimize(train_data, validation_data):
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    tuner = kt.Hyperband(build_model_hyperband, objective='val_acc', max_epochs=35, seed=13, directory='hyperband', project_name='test2')
    print(tuner.results_summary())
    # tuner.search(train_data, epochs=35, steps_per_epoch=300, validation_data=validation_data, callbacks=callbacks)
def optimize_bayes(train_data, validation_data):
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    tuner = kt.BayesianOptimization(build_model_bayes, objective='val_acc', max_trials=5, seed=13, directory='bayes',
                         project_name='test2')
    tuner.search(train_data, epochs=35, steps_per_epoch=300, validation_data=validation_data, callbacks=callbacks)

datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, rotation_range=15, brightness_range=[0.7, 1.3], zoom_range = [1, 1.2])
train_datagen = datagen.flow_from_directory('../English/Fnt/', subset='training', target_size=(14, 14),
                                            class_mode='categorical', batch_size=128, color_mode='grayscale')
validation_datagen = datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                 class_mode='categorical', batch_size=128, color_mode='grayscale')
optimize_bayes(train_datagen, validation_datagen)