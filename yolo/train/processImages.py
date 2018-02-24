

import numpy as np
import pandas as pd
import os,sys
import cv2
import pickle

ROOTDIR = os.path.expanduser('~/workspace/deepWildCount/')
image_dir = ROOTDIR + '/data/2015/'
train_dir = os.path.realpath('./train_images/')

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
w_train = pd.read_csv(allfile)

train_images = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')

width=7360
height=4912

im_size=416 #size of training imageas for yolo

nx = width//im_size
ny = height//im_size

wb_size=64 #size of bounding boxes we're going to create
sz_2=wb_size//2


all_imgs = []
for imagename in train_images: 
    im = cv2.imread(image_dir + imagename + '.JPG')
    df = w_train[w_train['image_name']==imagename]

    n_count=0
    for x in np.arange(0,width-im_size,im_size):
        for y in np.arange(0,height-im_size,im_size):
            img_data = {'object':[]}
            save_name = train_dir + '/' + imagename + '-' + str(n_count) + '.JPG'
            img = im[y:y+im_size,x:x+im_size,:]
            cv2.imwrite(save_name, img)
            img_data['filename'] = save_name
            img_data['width'] = im_size
            img_data['height'] = im_size
            n_count+=1
            thisDF = df[(df['xcoord'] > x) & (df['xcoord'] < x+im_size) & (df['ycoord'] > y) & (df['ycoord'] < y+im_size)]

            for i,point in thisDF.iterrows():
                obj = {}

                obj['name'] = 'wildebeest'

                xmin = max(point['xcoord'] - x - sz_2,0)
                xmax = min(point['xcoord'] - x + sz_2,im_size)
                ymin = max(point['ycoord'] - y - sz_2,0)
                ymax = min(point['ycoord'] - y + sz_2,im_size)
                obj['xmin'] = int(xmin)
                obj['ymin'] = int(ymin)
                obj['xmax'] = int(xmax)
                obj['ymax'] = int(ymax)
                img_data['object'] += [obj]

            all_imgs += [img_data]


#print(all_imgs)
with open(train_dir + '/annotations.pickle', 'wb') as handle:
   pickle.dump(all_imgs, handle)
                        

