#from bilinearupsampling import BilinearUpSampling2D
from keras.layers import Activation, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers import *
from keras.applications.vgg16 import VGG16
import tensorflow as tf


num_classes=2
# size of the region that will be classified
im_sz = 72
# size of final conv layer after 3 layes of max pooling
fc1_sz=im_sz//2//2//2


def getModel() -> Sequential:
    input_shape = (im_sz,im_sz, 3)

    model = Sequential()
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu',  name='conv2',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (3, 3), activation='relu',  name='conv3',padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu',  name='conv4',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(192, (3, 3), activation='relu',  name='conv5',padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu',  name='conv6',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (fc1_sz, fc1_sz), activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='predictions'))
    model.add(Flatten())
    return model

def getSegModel(input_width, input_height) -> Model:
    input_shape = (input_height,input_width,3)
    input_img = Input(shape=input_shape)
 
    x=Conv2D(48, (3, 3), activation='relu', name='conv1',padding='SAME')(input_img)
    x=Conv2D(48, (3, 3), activation='relu', name='conv2',padding='same')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(96, (3, 3), activation='relu', name='conv3',padding='same')(x)
    x=Conv2D(96, (3, 3), activation='relu', name='conv4',padding='same')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(192, (3, 3),activation='relu', name='conv5', padding='same')(x)
    x=Conv2D(192, (3, 3),activation='relu', name='conv6', padding='same')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(512, (fc1_sz, fc1_sz),activation='relu', padding='same',name='fc1')(x)
    x=Dropout(0.5)(x)
    x=Conv2D(256, (1, 1),activation='relu', padding='same',name='fc2')(x)
    x=Dropout(0.5)(x)
    x=Conv2D(num_classes, (1, 1),activation='softmax', padding='same',name='predictions')(x)
    x = BilinearUpSampling2D(target_size=(input_height,input_width))(x)
    model = Model(input_img, x)
    return model

def getVgg16SegModel(input_width, input_height) -> Sequential:
    input_shape = (input_height,input_width,3)
    input_img = Input(shape=input_shape)

    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape=input_shape)
    model=Sequential()
    for layer in base_model.layers:
        model.add(layer)
    model.add(Conv2D(256, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='predictions'))
#    model.add(BilinearUpSampling2D(target_size=(input_height,input_width)))
# add final bilinear interpolation layer
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [input_height,input_width])
    model.add(Lambda(resize_bilinear))

    for layer in model.layers[:-4]:
        layer.trainable = False
    return model

