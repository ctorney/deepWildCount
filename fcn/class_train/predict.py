
import tensorflow as tf

import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout, Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers

import numpy as np
import random
import cv2

ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'
img = cv2.imread(image_dir + 'SWC0002.JPG')


img = img[:512,:512,:]
im = img.astype('float32')/255
ny = im.shape[0]
nx = im.shape[1]

print(im.shape)
num_classes = 2

# load base model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=im.shape) 

fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)


fcn_model.add(Conv2D(256, (1,1), activation='relu', name='fc1',padding='VALID',input_shape=base_model.output_shape[1:]))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='predictions'))

#load classifier weights
fcn_model.load_weights('../weights/vgg16-cls.h5')

# add final bilinear interpolation layer
def resize_bilinear(images):
    return tf.image.resize_bilinear(images, [nx,ny])
fcn_model.add(Lambda(resize_bilinear))


print(fcn_model.summary())
preds = fcn_model.predict(im[None,:])
output = np.argmax(preds[0,:,:,:],2)
output = preds[0,:,:,1]
outRGB = cv2.cvtColor((255*output).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.imwrite('test_out.png',outRGB)
cv2.imwrite('test_in.png',img)
