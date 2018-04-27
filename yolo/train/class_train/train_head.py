
import time,os,sys
from keras.preprocessing import image
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
sys.path.append("..")
sys.path.append("../..")
from yolo_models import get_darknet_19
import numpy as np
import random
import cv2

num_classes = 2

im_sz = 96
# set-up the model
darknet = get_darknet_19(im_sz,im_sz,num_classes)

darknet.load_weights('../../weights/weights_coco.h5', by_name=True)
for layer in darknet.layers:
    layer.trainable = False
layer  = darknet.layers[-1] 
layer.trainable = True

print(darknet.summary())

optimizer = SGD(lr=1e-3, decay=0.0005, momentum=0.9)
darknet.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])


#create training samples
train_images = []
train_labels = []

import glob

# use 10000 images
nw=0
negatives =  glob.glob('train_images/nw/0/*.jpg')
random.shuffle(negatives)

for filename in negatives:
    if nw>10000: break
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
   

print(len(train_images))
print(len(train_labels))

perm = list(range(len(train_images)))
random.shuffle(perm)
train_images = [train_images[index] for index in perm]
train_labels = [train_labels[index] for index in perm]

testset = int(0.1*len(train_images))

x_train = np.asarray(train_images) #[testset:,:])
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

datagen = ImageDataGenerator(zoom_range=0.05, vertical_flip=False, horizontal_flip=False)


callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1)]

batch=1024
steps = len(x_train)//batch

# train the model
start = time.time()
print('Training starts')
model_info = darknet.fit_generator(datagen.flow(x_train, y_train, batch_size = batch),
                                 steps_per_epoch = steps, epochs = 200, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

darknet.save_weights('../../weights/darknet19-header.h5')
print('Total time taken:' + str(end - start))

