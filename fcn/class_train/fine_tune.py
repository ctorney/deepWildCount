
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
import numpy as np
import random
import cv2


TRAIN_STAGE=1

num_classes = 2

im_sz = 96
# set-up the model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(im_sz,im_sz,3) )

fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)

fcn_model.add(Cropping2D(cropping=((1, 1), (1, 1))))
fcn_model.add(Conv2D(256, (1,1), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='predictions'))
fcn_model.add(Flatten())

if TRAIN_STAGE>0:
    fcn_model.load_weights('../weights/vgg16-cls.h5')
else:
    fcn_model.load_weights('../weights/vgg16-header.h5')


# freeze the lower layers
for layer in fcn_model.layers[:15]:
    layer.trainable = False

print(fcn_model.summary())

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
fcn_model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])


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

testset = int(0.4*len(train_images))

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

datagen = ImageDataGenerator(zoom_range=0.0, vertical_flip=True, horizontal_flip=True)


callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1)]

# train the model
start = time.time()
print('Training starts')
model_info = fcn_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 1024),
                                 steps_per_epoch = 512, epochs = 10, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

fcn_model.save_weights('../weights/vgg16-cls.h5')
print('Total time taken:' + str(end - start))

