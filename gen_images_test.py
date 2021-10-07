#####################################################
# Converts MNIST numpy arrays to image files
# for calibration during quantization
#####################################################

import os
import shutil
import cv2
import warnings
from tensorflow.keras.datasets import mnist

#####################################################
# Set up directories
#####################################################

SCRIPT_DIR = os.getcwd()
IMG_DIR = os.path.join(SCRIPT_DIR, 'test_images')

classes = ['zero','one','two','three','four','five','six','seven','eight','nine']

if (os.path.exists(IMG_DIR)):
    shutil.rmtree(IMG_DIR)
os.makedirs(IMG_DIR)
print('Directory', IMG_DIR, 'created') 



#####################################################
# Get the dataset using Keras
#####################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#####################################################
# convert test dataset into image files
#####################################################

for i in range(len(x_test)):
    cv2.imwrite(os.path.join(IMG_DIR,'img_'+str(classes[y_test[i]])+'_'+str(i)+'.png'), x_test[i])


print ('FINISHED GENERATING TEST IMAGES')

