
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
sys.path.append("../..")
from yolo_models import get_darknet_19
import numpy as np
import random
import cv2


TRAIN_STAGE=0

num_classes = 2

im_sz = 96
# set-up the model
darknet = get_darknet_19(im_sz,im_sz,num_classes)

if TRAIN_STAGE>0:
    darknet.load_weights('../../weights/darknet-cls.h5')
else:
    darknet.load_weights('../../weights/darknet-header.h5')

optimizer = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
darknet.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

print(darknet.summary())


train_images = []
train_labels = []

import glob

# use 40000 images
nw=0
if TRAIN_STAGE:
    negatives =  glob.glob('train_images/nw/0/*.jpg')+glob.glob('cls_train_images_96/nw/1/*.jpg')
else:
    negatives =  glob.glob('train_images/nw/0/*.jpg')
random.shuffle(negatives)

for filename in negatives:
    if nw>40000: break
    img = cv2.imread(filename)
    if img is not None:
        train_images.append(np.array(img))
        train_labels.append(0)
        nw+=1


for filename in glob.iglob('train_images/w/*.jpg'):
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

y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

datagen = ImageDataGenerator(zoom_range=0.05, vertical_flip=True, horizontal_flip=True)


callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1)]

# train the model
start = time.time()
print('Training starts')
model_info = darknet.fit_generator(datagen.flow(x_train, y_train, batch_size = 1024),
                                 steps_per_epoch = 512, epochs = 10, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

darknet.save_weights('../../weights/darknet-cls.h5')
print('Total time taken:' + str(end - start))

