import numpy as np
import pandas as pd
import os,sys
import cv2
import pickle
sys.path.append("../..") 
from models.yolo_models import get_yolo_coco
from decoder import decode

ROOTDIR = os.path.expanduser('~/workspace/deepWildCount/')
image_dir = ROOTDIR + '/data/2015/'
train_dir = os.path.realpath('../train_images/')

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
w_train = pd.read_csv(allfile)

train_images = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')

width=7360
height=4912

im_size=416 #size of training imageas for yolo
im_size=864 #size of training imageas for yolo

nx = width//im_size
ny = height//im_size

wb_size=64 #size of bounding boxes we're going to create
sz_2=wb_size//2
yolov3 = get_yolo_coco(im_size,im_size)
print(yolov3.summary())
sys.exit('bye')
yolov3.load_weights('../../weights/yolo-v3-coco.h5')
im_num=1
all_imgs = []
for imagename in train_images: 
    im = cv2.imread(image_dir + imagename + '.JPG')
    df = w_train[w_train['image_name']==imagename]
    print('processing image ' + imagename + ', ' + str(im_num) + ' of 500 ...')
    im_num+=1

    n_count=0
    for x in np.arange(0,width-im_size,im_size):
        for y in np.arange(0,height-im_size,im_size):
            img_data = {'object':[]}
            save_name = train_dir + '/' + imagename + '-' + str(n_count) + '.JPG'
            box_name = train_dir + '/bbox/' + imagename + '-' + str(n_count) + '.JPG'
            img = im[y:y+im_size,x:x+im_size,:]
            cv2.imwrite(save_name, img)
            img_data['filename'] = save_name
            img_data['width'] = im_size
            img_data['height'] = im_size
            n_count+=1
            thisDF = df[(df['xcoord'] > x) & (df['xcoord'] < x+im_size) & (df['ycoord'] > y) & (df['ycoord'] < y+im_size)]
            # make the yolov3 model to predict 80 classes on COCO

            # preprocess the image
            image_h, image_w, _ = img.shape
            new_image = img[:,:,::-1]/255.
            new_image = np.expand_dims(new_image, 0)

            # run the prediction
            yolos = yolov3.predict(new_image)
            xloc = thisDF['xcoord'].values - x
            yloc = thisDF['ycoord'].values - y

            boxes = decode(yolos, im_size, xloc, yloc, box_name, img)
            for b in boxes:
                xmin=b[0]
                xmax=b[2]
                ymin=b[1]
                ymax=b[3]
                obj = {}

                obj['name'] = 'wildebeest'

 #               xmin = point['xcoord'] - x - sz_2
 #               xmax = point['xcoord'] - x + sz_2
 #               ymin = point['ycoord'] - y - sz_2
 #               ymax = point['ycoord'] - y + sz_2

                if xmin<0: continue
                if ymin<0: continue
                if xmax>im_size: continue
                if ymax>im_size: continue
                obj['xmin'] = int(xmin)
                obj['ymin'] = int(ymin)
                obj['xmax'] = int(xmax)
                obj['ymax'] = int(ymax)
                img_data['object'] += [obj]

            all_imgs += [img_data]


#print(all_imgs)
with open(train_dir + '/annotations.pickle', 'wb') as handle:
   pickle.dump(all_imgs, handle)
                        

