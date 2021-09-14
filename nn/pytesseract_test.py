# from tensorflow.keras import models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pytesseract
import cv2.cv2 as cv2
from PIL import Image
import numpy as np
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
image = cv2.imread('../aaai/015.png')
print(pytesseract.image_to_data(image, config='--psm 12 --oem 0'))
def compare():
    num_right = 0
    total_files = 0
#    model = models.load_model('bayes_optimized_character_model.h5')
    for root, dirs, files in os.walk("../English/Fnt/Sample001", topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (14, 14), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LINEAR)
            new_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

            img = new_image
            cv2.imshow('new_image', new_image)
            image_line = np.zeros(shape=(img.shape[0], img.shape[1]*10))
            for i in range(10):
                image_line[:, i*20:(i+1)*20] = img
            cv2.waitKey()
            print(pytesseract.image_to_string(Image.fromarray(image_line.astype(np.uint8)), config=' --psm 7'))


            total_files = total_files + 1
            if(total_files%10==0):
                print(total_files, num_right)
