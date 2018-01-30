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

import numpy as np
import random
import cv2

num_classes = 2

# initialize deconv layer as bilinear interpolation
def get_deconv_weights(f_shape):
    width = f_shape[0]
    heigh = f_shape[1]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]],dtype=np.float32)
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape,dtype=np.float32)
    for i in range(f_shape[2]):
        weights[:, :,i,i] = bilinear

    return weights
 
# baseline fcn model - no skip layers
def fcn_32s_model():
    input_shape = (512,512,3)
    num_classes = 2

    # load base model
    base_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape) 

    ip = Input(shape=input_shape)
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

    h = base_model.layers[11](h)
    h = base_model.layers[12](h)
    h = base_model.layers[13](h)
    h = base_model.layers[14](h)

    h = base_model.layers[15](h)
    h = base_model.layers[16](h)
    h = base_model.layers[17](h)
    h = base_model.layers[18](h)

    h = Conv2D(256, (1,1), activation='relu', name='fc1',padding='VALID')(h)


    h = Conv2D(num_classes, (1, 1), activation='relu', name='predictions')(h)

    h = Conv2DTranspose(num_classes, (8, 8), strides=(4, 4), padding='valid',name='p5_deconv')(h)
    h = Cropping2D(((2, 2), (2, 2)),name='final_low_res')(h)

    h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),padding='valid',name='final_deconv')(h)
    h = Cropping2D(((4, 4), (4, 4)))(h)

    h = Activation('softmax')(h)
    fcn32s = Model(ip,h)
    # load weights from pretrained classifier
    fcn32s.load_weights('../weights/vgg16-cls.h5',by_name=True)

    # add the weights for the transpose layers so they approximate bilinear upsampling
    w1_ = np.zeros([2],dtype=np.float32)

    w_ = get_deconv_weights([8,8,num_classes, num_classes])
    fcn32s.get_layer('p5_deconv').set_weights([w_,w1_])
    w_ = get_deconv_weights([16,16,num_classes, num_classes])
    fcn32s.get_layer('final_deconv').set_weights([w_,w1_])

    # freeze everything but the top layers
    for layer in fcn32s.layers[:-7]:
        layer.trainable = False

    if os.path.isfile('fcn_32s_weights.h5'):
        fcn32s.load_weights('fcn_32s_weights.h5')

    return fcn32s

def fcn_16s_model():
    input_shape = (512,512,3)

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

#p4 is input //16
    #p4 = Conv2D(num_classes, (1, 1), activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros', padding='valid',name='p4_conv')(p4)
    p4 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid',name='p4_conv')(p4)

    p4 = Conv2DTranspose(num_classes, (4, 4),strides=(2, 2),padding='valid',name='p4_deconv')(p4)
    p4 = Cropping2D(((1, 1), (1, 1)))(p4)


#p5 is input//32
    p5 = Conv2D(num_classes, (1, 1), activation='relu', name='predictions')(p5)

    p5 = Conv2DTranspose(num_classes, (8, 8), strides=(4, 4), padding='valid',name='p5_deconv')(p5)
    p5 = Cropping2D(((2, 2), (2, 2)),name='final_low_res')(p5)

    h = add([p4, p5])
    h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),padding='valid',name='final_deconv')(h)
    #h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),padding='valid',name='final_deconv')(p5)
    h = Cropping2D(((4, 4), (4, 4)))(h)
    h = Activation('softmax')(h)

#h = Softmax2D()(h)
    fcn16s = Model(ip,h)
    fcn16s.load_weights('../weights/vgg16-cls.h5',by_name=True)
# freeze the lower layers
    for layer in fcn16s.layers[:19]:
        layer.trainable = False
    w_ = get_deconv_weights([4,4,num_classes, num_classes])
    w1_ = np.zeros([2],dtype=np.float32)
    fcn16s.get_layer('p4_deconv').set_weights([w_,w1_])
    w_ = get_deconv_weights([8,8,num_classes, num_classes])
    fcn16s.get_layer('p5_deconv').set_weights([w_,w1_])
    w_ = get_deconv_weights([16,16,num_classes, num_classes])
 #   fcn16s.get_layer('p4_conv').set_weights(l4_model().get_layer('p4_conv').get_weights())

    fcn16s.get_layer('p4_conv').trainable=True
    fcn16s.get_layer('p4_deconv').trainable=True
    fcn16s.get_layer('p5_deconv').trainable=True
    fcn16s.get_layer('final_deconv').trainable=True
    if os.path.isfile('fcn_32s_weights.h5'):
        fcn16s.load_weights('fcn_32s_weights.h5',by_name=True)
    if os.path.isfile('fcn_16s_weights.h5'):
        fcn16s.load_weights('fcn_16s_weights.h5')
    return fcn16s

def fcn_8s_model():
    input_shape = (512,512,3)

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
    #p3 = Conv2D(num_classes, (1, 1), activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros', padding='valid',name='p3_conv')(p3)
    p3 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid',name='p3_conv')(p3)
#p4 is input //16
    #p4 = Conv2D(num_classes, (1, 1), activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros', padding='valid',name='p4_conv')(p4)
    p4 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid',name='p4_conv')(p4)

    p4 = Conv2DTranspose(num_classes, (4, 4),strides=(2, 2),padding='valid',name='p4_deconv')(p4)
    p4 = Cropping2D(((1, 1), (1, 1)))(p4)


#p5 is input//32
    p5 = Conv2D(num_classes, (1, 1), activation='relu', name='predictions')(p5)

    p5 = Conv2DTranspose(num_classes, (8, 8), strides=(4, 4), padding='valid',name='p5_deconv')(p5)
    p5 = Cropping2D(((2, 2), (2, 2)),name='final_low_res')(p5)

    h = add([p3, p4, p5])
    h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),padding='valid',name='final_deconv')(h)
    #h = Conv2DTranspose(num_classes, (16, 16),strides=(8, 8),padding='valid',name='final_deconv')(p5)
    h = Cropping2D(((4, 4), (4, 4)))(h)
    h = Activation('softmax')(h)

#h = Softmax2D()(h)
    fcn8s = Model(ip,h)
    fcn8s.load_weights('../weights/vgg16-cls.h5',by_name=True)
# freeze the lower layers
    for layer in fcn8s.layers: #[:15]:
        layer.trainable = False
    w_ = get_deconv_weights([4,4,num_classes, num_classes])
    w1_ = np.zeros([2],dtype=np.float32)
    fcn8s.get_layer('p4_deconv').set_weights([w_,w1_])
    w_ = get_deconv_weights([8,8,num_classes, num_classes])
    fcn8s.get_layer('p5_deconv').set_weights([w_,w1_])
    w_ = get_deconv_weights([16,16,num_classes, num_classes])
    fcn8s.get_layer('p3_conv').set_weights(l3_model().get_layer('p3_conv').get_weights())
    fcn8s.get_layer('p4_conv').set_weights(l4_model().get_layer('p4_conv').get_weights())

    fcn8s.get_layer('p3_conv').trainable=True
    fcn8s.get_layer('p4_conv').trainable=True
    fcn8s.get_layer('p4_deconv').trainable=True
    fcn8s.get_layer('p5_deconv').trainable=True
    fcn8s.get_layer('final_deconv').trainable=True
    if os.path.isfile('fcn_32s_weights.h5'):
        fcn16s.load_weights('fcn_32s_weights.h5',by_name=True)
    if os.path.isfile('fcn_16s_weights.h5'):
        fcn16s.load_weights('fcn_16s_weights.h5',by_name=True)
    if os.path.isfile('fcn_8s_weights.h5'):
        fcn16s.load_weights('fcn_8s_weights.h5')
    return fcn8s

#
def l3_model():
# get lower levels model for pretrained conv3 layer
    input_shape = (96,96,3)
    full_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape )

    ip = Input(shape=input_shape) #(3, self.img_height, self.img_width))
    h = full_model.layers[1](ip)
    h = full_model.layers[2](h)
    h = full_model.layers[3](h)
    h = full_model.layers[4](h)
    h = full_model.layers[5](h)
    h = full_model.layers[6](h)
    h = full_model.layers[7](h)
    h = full_model.layers[8](h)
    h = full_model.layers[9](h)
    h = full_model.layers[10](h)

    h = Conv2D(num_classes, (1, 1), activation='relu', padding='valid',name='p3_conv')(h)
    h = Cropping2D(cropping=((5, 6), (5, 6)))(h)
    h = Flatten()(h)
    l3_model = Model(ip,h)
    l3_model.load_weights('../weights/vgg16-layer-3.h5')
    return l3_model
def l4_model():
# get lower levels model for pretrained conv4 layer
    input_shape = (96,96,3)
    full_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape )

    ip = Input(shape=input_shape) #(3, self.img_height, self.img_width))
    h = full_model.layers[1](ip)
    h = full_model.layers[2](h)
    h = full_model.layers[3](h)
    h = full_model.layers[4](h)
    h = full_model.layers[5](h)
    h = full_model.layers[6](h)
    h = full_model.layers[7](h)
    h = full_model.layers[8](h)
    h = full_model.layers[9](h)
    h = full_model.layers[10](h)
    h = full_model.layers[11](h)
    h = full_model.layers[12](h)
    h = full_model.layers[13](h)
    h = full_model.layers[14](h)

    h = Conv2D(num_classes, (1, 1), activation='relu', padding='valid',name='p4_conv')(h)
    h = Cropping2D(cropping=((2, 3), (2, 3)))(h)
    h = Flatten()(h)
    l4_model = Model(ip,h)
    l4_model.load_weights('../weights/vgg16-layer-4.h5')
    return l4_model


