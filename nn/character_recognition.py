from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import Resizing

datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, rotation_range=15, brightness_range=[0.7, 1.3], zoom_range = [1, 1.2])
train_datagen = datagen.flow_from_directory('../English/Fnt/', subset='training', target_size=(14, 14),
                                            class_mode='categorical', batch_size=128, color_mode='grayscale')
validation_datagen = datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                 class_mode='categorical', batch_size=128, color_mode='grayscale')
model = models.Sequential()
model.add(Resizing(46, 46))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.4))
model.add(layers.Dense(62, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
file_model = models.load_model('big_image_model.h5')
print(file_model.summary())
model.fit(train_datagen, epochs=35, steps_per_epoch=300, validation_data=validation_datagen)
