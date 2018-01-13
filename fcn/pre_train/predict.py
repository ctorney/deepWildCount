

import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
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
base_model = VGG16(weights=None,include_top=False,input_shape=(64,64,3) )

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


fcn_model.add(Conv2D(4096, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2'))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='predictions'))
fcn_model.add(Flatten())



fcn_model.load_weights('vgg19-tiny-imagenet.h5')
# CONCATENATE THE TWO MODELS
#new_model.add(model3)
print(fcn_model.summary())

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = fcn_model.predict(x)
for i in range(200): print(preds[0,i])
print(preds.shape)
res = np.argmax(preds,1)
import glob
label_counter = 0 

for imdir in os.listdir('tiny-imagenet-200/train/'):
    if label_counter==res: print(imdir)
    if imdir =='n02504458': print(label_counter)
    label_counter = label_counter + 1
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])

