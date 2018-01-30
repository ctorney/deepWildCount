
import numpy as np
import pandas as pd
import os
import cv2
import random


ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'
train_dir = 'train_images/nw/0/'

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
w_train = pd.read_csv(allfile)

train_images = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')

sample_count=10000*4

train_count = len(train_images)

dist = np.zeros(train_count,dtype='int')

# distribute sample between training images
for i in range(sample_count):
    k = random.randint(0,train_count-1)
    dist[k]+=1

im_size=96
sz_2=im_size//2

# closest point to a wildebeest we'll accept (squared to save rooting the distance)
min_dist = 32**2

for i in range(train_count):

    imagename = train_images[i]
    im = cv2.imread(image_dir + imagename + '.JPG')
    df = w_train[w_train['image_name']==imagename]
    x_w = df['xcoord'].values
    y_w = df['ycoord'].values
    # dist stores how many samples we take from this image
    for j in range(dist[i]):
        while(1):
            x = random.randint(sz_2,7360-sz_2)
            y = random.randint(32,4912-sz_2)
            closest = 2*min_dist
            if len(x_w):
                distances = (x-x_w)**2 + (y-y_w)**2
                closest = np.min(distances)
            if closest>min_dist:
                img = im[y-sz_2:y+sz_2,x-sz_2:x+sz_2,:]

                if img.shape == (im_size,im_size,3):
                    cv2.imwrite(train_dir + imagename + '_' + str(j) + '.jpg', img)
                    break

#w_train = w_train[(w_train['xcoord'] > 32) & (w_train['xcoord'] < 7325) & (w_train['ycoord'] > 32) & (w_train['ycoord'] < 4879)]

#for imagename in train_images: 
#    im = cv2.imread(image_dir + imagename + '.JPG')
#    df = w_train[w_train['image_name']==imagename]

#    for i,point in df.iterrows():


#        img = im[int(point['ycoord']) - sz_2:int(point['ycoord'])+sz_2,int(point['xcoord']) - sz_2:int(point['xcoord'])+sz_2,:]

#        if img.shape == (im_size,im_size,3):
#            cv2.imwrite(train_dir + imagename + '_' + str(i) + '.jpg', img)

