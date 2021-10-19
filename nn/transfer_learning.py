import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
import numpy as np
import cv2.cv2 as cv2










def get_bottleneck_features_and_labels():
    train_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)
    train_data = train_datagen.flow_from_directory('../English/Fnt/', subset='training', target_size=(128, 128),
                                                   class_mode='categorical', batch_size=128, seed=42, shuffle=False)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    validation_data = test_datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(128, 128),
                                                       class_mode='categorical', batch_size=128, seed=42, shuffle=False)
    base = ResNet50(include_top=False, input_shape=(128, 128, 3))
    base.trainable = False
    inputs = keras.Input(shape=(128, 128, 3))
    # x = Resizing(50, 50, input_shape=(14, 14))(inputs)
    x = base(inputs, training=False)
    model = keras.Model(inputs, x)
    num_train = len(train_data.filenames)
    num_validation = len(validation_data.filenames)
    train_features = model.predict(train_data, num_train, verbose=1)
    validation_features = model.predict(validation_data, num_validation, verbose=1)
    train_labels = to_categorical(train_data.classes, num_classes=62)
    validation_labels = to_categorical(validation_data.classes, num_classes=62)
    np.save('transfer_learning/resnet_features_max_size.np', train_features)
    np.save('transfer_learning/resnet_validation_features_max_size.np', validation_features)
    return train_features, validation_features, train_labels, validation_labels
def build_top(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(rate=0.3))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(62, activation='softmax'))
    return model


# train, val, train_labels, val_labels = get_bottleneck_features_and_labels()
# top = build_top(input_shape=train.shape[1:])
# top.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# top.fit(x=train, y=train_labels, batch_size=128, epochs=50, validation_data=(val, val_labels))

# image = cv2.imread('../English/Fnt/Sample029/img029-00003.png')
# resized = cv2.resize(image, (14, 14))
# cv2.imwrite('resized_image.png', resized)



upscaler = cv2.dnn.DNN