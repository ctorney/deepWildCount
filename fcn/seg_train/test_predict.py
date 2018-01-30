import tensorflow as tf
import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout, Lambda
from keras.layers import Conv2D, Deconvolution2D, Cropping2D, Activation
from keras.layers import MaxPooling2D, Conv2DTranspose, add, Input
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers
from math import ceil

from fcn_networks import *
import numpy as np
import random
import cv2
num_classes = 2

mfcn8s = fcn_16s_model()

print(mfcn8s.summary())

ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'
img = cv2.imread(image_dir + 'SWC1077.JPG')


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


layer_name = 'final_low_res'
#layer_name = 'predictions'
i_model = Model(inputs=mfcn8s.input,outputs=mfcn8s.get_layer(layer_name).output)
##intermediate_output = intermediate_layer_model.predict(data)
#print(i_model.summary())
#
# add final bilinear interpolation layer
#def resize_bilinear(images):
#    return tf.image.resize_bilinear(images, [nx,ny])
#fcn_model.add(Lambda(resize_bilinear))

#print(fcn_model.summary())
preds = fcn_model.predict(im[None,:])
output = np.argmax(preds[0,:,:,:],2)
#output = preds[0,:,:,1]
outRGB = cv2.cvtColor((255*output).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.imwrite('test_out.png',outRGB)
cv2.imwrite('test_in.png',img)
preds = mfcn8s.predict(im[None,:])
#preds = i_model.predict(im[None,:])
output = np.argmax(preds[0,:,:,:],2)
output = preds[0,:,:,1]
outRGB = cv2.cvtColor((255*output).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.imwrite('test_out8s.png',outRGB)
cv2.imwrite('test_in8s.png',img)
img = cv2.imread(image_dir + 'labels/SWC1077.png')
img = img[:512,:512,:]
cv2.imwrite('test_label.png',img)


import math
import keras.backend as K
def binary_crossentropy(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return np.mean(result)
img = np.array(img)[:, : , 0]
            #print(np.sum(img[:,:]))
            #correct for pixel intensity
input_ = np.around((img * (2 - 1)) / 255.0)
print(binary_crossentropy(input_,output))


#for p in input_:
#    print(p)
