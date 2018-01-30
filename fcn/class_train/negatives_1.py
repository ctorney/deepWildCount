import time,os,sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical

import tensorflow as tf

import time,os,sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input, Dropout, Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers

from models import simpleCNN
import time

ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'
train_dir = 'train_images/nw/1/'

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
trainfile = ROOTDIR + '/data/2015-checked-train.txt'

np.random.seed(2017)
nx = 512
ny = 512

cls_im_sz = 96
im2 = cls_im_sz//2
#fcnmodel = simpleCNN.getVgg16SegModel(ny,nx)
num_classes = 2

# load base model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(ny,nx,3)) 

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
#num_classes=2

#fcnmodel.load_weights('weights/vgg16-cls.h5')
train_images = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
df2 = pd.read_csv(allfile)
#df2 = pd.read_csv('2015-Z-LOCATIONS.csv')
not_wildebeest = pd.DataFrame(columns=['image_name', 'xcoord', 'ycoord'])
not_wildebeest.to_csv(ROOTDIR + '/data/new_not_wildebeest_locations.csv', header=True, mode='w')

i=0
for filename in train_images: 
    start = time.time()
    x_pos, y_pos = 0,0
 #   filename = df.iloc[i][0]
    print('Examining image ' + str(i+1) + ' of ' + str(len(train_images)) +  ': ' + filename)
    i=i+1
    img = cv2.imread(image_dir + str(filename) + '.JPG')
    df3 = df2.loc[df2.image_name == filename]
    w_xpos = df3.xcoord.values
    w_xpos = df2[df2.image_name == filename].xcoord.values
    w_ypos = df3.ycoord.values
    # imarr = np.zeros((img.size[0], img.size[1], 3))
    filenames = []
    x_ts = []
    y_ts = []
    j = 0
    for y_pos in range(0, img.shape[0], ny):
        for x_pos in range(0, img.shape[1], nx):
            #arr = np.array(img)
            arr = img[y_pos:ny+y_pos,x_pos:nx+x_pos,:]
            arry, arrx = arr.shape[0], arr.shape[1]
            if arrx < nx:
                continue
                arr2 = np.zeros((nx - arrx, arry, 3))
                arr = np.concatenate((arr, arr2), axis=0)
            if arry < ny:
                continue
                arr2 = np.zeros((arr.shape[0], ny - arry, 3))
                arr = np.concatenate((arr, arr2), axis=1)
            #arr = arr.astype('float32')/255.0
            X = []
            X.append(arr)
            X = np.asarray(X)
            X = X.astype('float32')/255
            result = fcn_model.predict(X)[0]
            predicted_class = np.argmax(result, axis=2)
            prediction = result[:,:,1]
#output = preds[0,:,:,1]
 #           outRGB = cv2.cvtColor((255*predicted_class).astype(np.uint8),cv2.COLOR_GRAY2BGR)
 #           cv2.imwrite('test_pred/test_' + filename + str(x_pos) + str(y_pos) + 'out.png',outRGB)
 #           cv2.imwrite('test_pred/test_' + filename + str(x_pos) + str(y_pos) + 'in.png',arr)
 #           outRGB = cv2.cvtColor((255*result[:,:,1]).astype(np.uint8),cv2.COLOR_GRAY2BGR)
  #          cv2.imwrite('test_out.png',outRGB)
   #         break

            for y_in in range(im2*3, ny-im2, im2):
                for x_in in range(im2, nx-im2, im2):
                    flag = False
                    #if predicted_class[y_in,x_in]:
                    if prediction[y_in,x_in]>0.8:
                        x_t, y_t = x_pos + x_in, y_pos + y_in 
                        # try loop
  #                      start = time.time()
  #                      for index, d in df3.iterrows():
  #                          if ((d.xcoord - x_t) ** 2) + ((d.ycoord - y_t) ** 2) <= 1600:
  #                              flag = True
  #                              break
  #                      end = time.time()
  #                      print(end - start)
 #                       flag2 = False
  #                      start = time.time()
                        if np.any(((w_xpos-x_t)**2+(w_ypos-y_t)**2)<=1600):
                            flag = True
  #                      end = time.time()
   #                     print(end - start)
    #                    print(flag,flag2)
                        if flag == False:
  #                          start = time.time()
                            filenames.append(filename)
                            x_ts.append(x_t)
                            y_ts.append(y_t)
   #                         end = time.time()
 #                           print(end - start)
 #                           if x_in >= 50 and y_in >= 50 and x_in <= nx - 50 and y_in <= ny - 50:
                            tiny_im = arr[y_in - im2:y_in+im2, x_in - im2:x_in+im2, :]
#                                tiny_img = Image.fromarray(np.uint8(tiny_im))
                            if tiny_im.shape == (cls_im_sz,cls_im_sz,3):
                                cv2.imwrite(train_dir + filename + '_' + str(j) + '.jpg', tiny_im)
                                j = j + 1
                            # not_wildebeest = not_wildebeest.append({'image_name':filename, 'xcoord':x_t, 'ycoord':y_t}, ignore_index=True)
            # predicted_class = predicated_class * 255.0
            # predicted_class = np.concatenate((predicted_class, predicted_class, predicted_class) axis=2)
            # imarr[x_pos:x_pos+arrx,y_pos:arry+y_pos),:] = predicted_class[0:arrx,0:arry, :]
    # img = Image.fromarray(np.uint8(im_arr))
    # img.save('2015/test/' + str(filename) + '.png')
    end = time.time()
    print('Image ' + filename + ' created ' + str(j) + ' misclassified samples in ' + str(end - start) + ' seconds')
    not_wildebeest = pd.DataFrame(np.column_stack([filenames,x_ts,y_ts]), columns=['image_name', 'xcoord', 'ycoord'])
    not_wildebeest.to_csv('test/new_not_wildebeest_locations.csv', header=False, mode='a')

print('Finished.')
