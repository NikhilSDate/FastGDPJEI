from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import kerastuner as kt
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
def build_model(hp):

    model = keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(hp.Int('filters', 32, 64, step=4), kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))




    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
batch_size = 128
epochs = 15
tuner = kt.Hyperband(build_model, objective='val_acc', hyperband_iterations=2, max_epochs=20)

tuner.search(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)