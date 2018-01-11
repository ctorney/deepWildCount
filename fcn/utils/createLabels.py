from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os

ROOTDIR = '../../'
raw_image_folder = ROOTDIR + '/data/2015/'
label_image_folder = raw_image_folder + '/labels/'

data_file_csv = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
count_data = pd.read_csv(data_file_csv)

#get a set of all unique image names in the data file
image_names = os.listdir(raw_image_folder)

rd = 5
#for each image, create an array to store our ground truth
for name in image_names:
    if name.endswith('.JPG'):

        fname=name[:7]

        #look at rows in the datafile corresponding to that image
        count_rows = count_data.loc[count_data['image_name'] == fname]

        img = Image.new(mode='RGB',size=(7360,4912),color=(0,0,0))
        draw = ImageDraw.Draw(img,mode='RGB')

        #for each row, increment the array around that location by 1
        for index, row in count_rows.iterrows():
            draw.ellipse((int(row.xcoord) - rd, int(row.ycoord) - rd, int(row.xcoord) + rd, int(row.ycoord) + rd), fill=(255,255,255))
        newfilename = label_image_folder + '/' + fname + '.png'
        img.save(newfilename)
