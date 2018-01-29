
import tensorflow as tf

import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout, Lambda
from keras.layers import Conv2D, Deconvolution2D, Cropping2D
from keras.layers import MaxPooling2D, Conv2DTranspose, add, Input
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers

import numpy as np
import random
import cv2

def fcn_8s_model():
    input_shape = (512,512,3)
    num_classes = 2

# load base model
    base_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape) 

    ip = Input(shape=input_shape) #(3, self.img_height, self.img_width))
    h = base_model.layers[1](ip)
    h = base_model.layers[2](h)
    h = base_model.layers[3](h)
    h = base_model.layers[4](h)
    h = base_model.layers[5](h)
    h = base_model.layers[6](h)
    h = base_model.layers[7](h)
    h = base_model.layers[8](h)
    h = base_model.layers[9](h)
    h = base_model.layers[10](h)

# split layer
    p3 = h

    h = base_model.layers[11](h)
    h = base_model.layers[12](h)
    h = base_model.layers[13](h)
    h = base_model.layers[14](h)

# split layer
    p4 = h
    h = base_model.layers[15](h)
    h = base_model.layers[16](h)
    h = base_model.layers[17](h)
    h = base_model.layers[18](h)

    h = Conv2D(256, (1,1), activation='relu', name='fc1',padding='VALID')(h)
    p5 = h

# get scores
#p3 is input //8
    p3 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid')(p3)
#p4 is input //16
    p4 = Conv2D(num_classes, (1, 1), activation='relu')(p4)

    p4 = Conv2DTranspose(num_classes, (4, 4),strides=(2, 2),padding='valid')(p4)
    p4 = Cropping2D(((1, 1), (1, 1)))(p4)


#p5 is input//32
    p5 = Conv2D(num_classes, (1, 1), activation='relu', name='predictions')(p5)

    p5 = Conv2DTranspose(num_classes, (8, 8), strides=(4, 4), padding='valid')(p5)
    p5 = Cropping2D(((2, 2), (2, 2)),name='final_low_res')(p5)

    h = add([p3, p4, p5])

    h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),border_mode='valid')(h)
    h = Cropping2D(((4, 4), (4, 4)))(h)

#h = Softmax2D()(h)
    fcn8s = Model(ip,h)
    fcn8s.load_weights('../weights/vgg16-cls.h5',by_name=True)
# freeze the lower layers
    for layer in fcn8s.layers[:15]:
        layer.trainable = False
    
    return fcn8s

#
mfcn8s = fcn_8s_model()

print(mfcn8s.summary())

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


layer_name = 'final_low_res'
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
##output = preds[0,:,:,1]
outRGB = cv2.cvtColor((255*output).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.imwrite('test_out.png',outRGB)
cv2.imwrite('test_in.png',img)
preds = i_model.predict(im[None,:])
output = np.argmax(preds[0,:,:,:],2)
##output = preds[0,:,:,1]
outRGB = cv2.cvtColor((255*output).astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.imwrite('test_out8s.png',outRGB)
cv2.imwrite('test_in8s.png',img)
