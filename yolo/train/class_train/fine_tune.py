
import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop
sys.path.append("../..")
from yolo_models import get_darknet_19
from yolo_models import get_yolo_cls
import tensorflow as tf
import numpy as np
import random
import cv2

def custom_loss(y_true, y_pred):
    gamma=0.
    #y_pred_in = tf.sigmoid(y_pred[..., 0]) + 1e-6
    y_pred_in = K.clip(y_pred[..., 0],  K.epsilon(),  1.-K.epsilon())
    y_true_in = y_true[..., 0]
    pt_1 = tf.where(tf.equal(y_true_in, 1), y_pred_in, tf.ones_like(y_pred_in))
    pt_0 = tf.where(tf.equal(y_true_in, 0), y_pred_in, tf.zeros_like(y_pred_in))
    loss = -K.mean(K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.mean(K.pow( pt_0, gamma) * K.log(1. - pt_0))

    """
    Debugging code
    """    
    #current_recall = nb_pred_box/(nb_true_box + 1e-6)
    #total_recall = tf.assign_add(total_recall, current_recall) 

#    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \n', summarize=1000)
#    loss = tf.Print(loss, [loss_fp], message='Binary loss \n', summarize=1000)
#    loss = tf.Print(loss, [loss_xy], message='xy loss \n', summarize=1000)
#    loss = tf.Print(loss, [pred_box_conf], message='prediction \n', summarize=1000)
#    loss = tf.Print(loss, [true_box_xy], message='true xy \n', summarize=1000)
    #loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    #loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    #loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    #loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    #loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    #loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    #loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
#    loss = tf.Print(loss, [tf.shape(y_pred)], message='shape \n', summarize=1000)
#    loss = tf.Print(loss, [tf.shape(y_true)], message='true shape \n', summarize=1000)
 #   if loss2>0:
    #loss = tf.Print(loss, [loss_ce], message='ce value \t', summarize=1000)
    #loss = tf.Print(loss, [(2*loss2)], message='fl value \t', summarize=1000)
    
    return loss


TRAIN_STAGE=1

num_classes = 2

im_sz = 96
# set-up the model
darknet = get_yolo_cls(im_sz,im_sz)

if TRAIN_STAGE>0:
    darknet.load_weights('../../weights/wb_yolo.h5',by_name=True)
else:
    darknet.load_weights('../../weights/darknet19-header.h5')

optimizer = SGD(lr=1e-3, decay=0.0005, momentum=0.9)
darknet.compile(optimizer=optimizer,loss=custom_loss,metrics=['accuracy'])
#darknet.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

print(darknet.summary())


train_images = []
train_labels = []

import glob

# use 40000 images
nw=0
w=0
if TRAIN_STAGE:
    negatives =  glob.glob('train_images/nw/0/*.jpg')+glob.glob('cls_train_images_96/nw/1/*.jpg')
else:
    negatives =  glob.glob('train_images/nw/0/*.jpg')
random.shuffle(negatives)

for filename in negatives:
 #   if nw>400: break
    img = cv2.imread(filename)
    if img is not None:
        train_images.append(np.array(img))
        train_labels.append(0)
        nw+=1


for filename in glob.iglob('train_images/w/*.jpg'):
  #  if w>400: break
    img = cv2.imread(filename)
    if img is not None:
        train_images.append(np.array(img))
        train_labels.append(1)
        rimg=cv2.flip(img,1)
        train_images.append(np.array(rimg))
        train_labels.append(1)
        uimg=cv2.flip(img,0)
        train_images.append(np.array(uimg))
        train_labels.append(1)
        luimg=cv2.flip(img,-1)
        train_images.append(np.array(luimg))
        train_labels.append(1)
        w+=1
   

perm = list(range(len(train_images)))
random.shuffle(perm)
train_images = [train_images[index] for index in perm]
train_labels = [train_labels[index] for index in perm]

testset = int(0.1*len(train_images))

x_train = np.asarray(train_images) #[testset:,:]
y_train = np.asarray(train_labels) #[testset:,:])
x_test = np.asarray(train_images) #[:testset,:])
y_test = np.asarray(train_labels) #[:testset,:])

x_train = x_train[testset:]
y_train = y_train[testset:]
x_test = x_test[:testset]
y_test = y_test[:testset]
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#y_train = to_categorical(y_train,num_classes)
#y_test = to_categorical(y_test,num_classes)

datagen = ImageDataGenerator(zoom_range=0.05, vertical_flip=False, horizontal_flip=False)


callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1)]
batch=128
steps = len(x_train)//batch

# train the model
start = time.time()
print('Training starts')
model_info = darknet.fit_generator(datagen.flow(x_train, y_train, batch_size = batch),
                                 steps_per_epoch = steps, epochs = 100, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

darknet.save_weights('../../weights/yolo_cls.h5')
print('Total time taken:' + str(end - start))

