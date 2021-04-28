from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2.cv2 as cv2
train_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, rotation_range=15, brightness_range=[0.7, 1.3], zoom_range = [1, 1.2])
train_data = train_datagen.flow_from_directory('../English/Fnt/', subset='training', target_size=(14, 14),
                                            class_mode='categorical', batch_size=64, color_mode='grayscale', seed=42)
test_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

validation_data = test_datagen.flow_from_directory('../English/Fnt/', subset='validation', target_size=(14, 14),
                                                 class_mode='categorical', batch_size=64, color_mode='grayscale', seed=42)

batch = next(validation_datagen)
for image in batch[0]:
    cv2.imshow('image', cv2.resize(image, (36, 36), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey()
# dataset = image_dataset_from_directory('../English/Fnt/')