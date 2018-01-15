
import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import random
import cv2

num_classes = 2

# set-up the model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(64,64,3) )

fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)

for layer in fcn_model.layers:
    layer.trainable = False

fcn_model.add(Conv2D(256, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='predictions'))
fcn_model.add(Flatten())

print(fcn_model.summary())

fcn_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


#create training samples
train_images = []
train_labels = []

import glob

# use 10000 images
nw=0
for filename in glob.iglob('cls_train_images/nw/0/*.jpg'):
    if nw>10000: break
    img = cv2.imread(filename)
    if img is not None:
        train_images.append(np.array(img))
        train_labels.append(0)
        nw+=1


for filename in glob.iglob('cls_train_images/w/*.jpg'):
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

datagen = ImageDataGenerator(zoom_range=0.0, vertical_flip=False, horizontal_flip=False)


callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1)]

# train the model
start = time.time()
print('Training starts')
model_info = fcn_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 1024),
                                 steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

fcn_model.save_weights('weights/vgg16-header.h5')
print('Total time taken:' + str(end - start))

