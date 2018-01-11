
import numpy as np
import pandas as pd
import cv2

ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'

movieList = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')

for imagename in movieList: 
    img1 = cv2.imread(image_dir + imagename + '.JPG')
    img2 = cv2.imread(image_dir + '/labels/' + imagename + '.png')

    #im = cv2.addWeighted(img1,0.7,img2,0.3,0)
    img1[img2>0]=255
    cv2.imwrite(image_dir + 'label_check/' + imagename + '.png',img1)


