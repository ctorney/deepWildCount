
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

num_classes = 200

#model = VGG16(weights='imagenet',include_top=True)
#print(model.summary())
#base_model = VGG19(weights='imagenet',include_top=False,input_shape=(64,64,3) )
#print(base_model.summary())
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(64,64,3) )

print(base_model.summary())
#sys.exit('bye')

# CREATE A TOP MODEL
#model3 = Sequential()
#model3.add(Flatten(


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)

for layer in fcn_model.layers:
    layer.trainable = False

fcn_model.add(Conv2D(4096, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2'))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='predictions'))
fcn_model.add(Flatten())

# CONCATENATE THE TWO MODELS
#new_model.add(model3)
#print(model.summary())
print(fcn_model.summary())

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
fcn_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


train_images = []
train_labels = []

label_counter=0

import glob

for imdir in os.listdir('tiny-imagenet-200/train/'):
    for filename in glob.iglob('tiny-imagenet-200/train/' + imdir +  '/images/*.JPEG'):
        img = cv2.imread(filename)
        if img is not None:
            train_images.append(np.array(img))
            train_labels.append(label_counter)
    label_counter = label_counter + 1

   

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


callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1)]

# train the model
start = time.time()
print('Training starts')
model_info = fcn_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 1024),
                                 steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

fcn_model.save_weights('vgg19-tiny-imagenet.h5')
print('Total time taken:' + str(end - start))

