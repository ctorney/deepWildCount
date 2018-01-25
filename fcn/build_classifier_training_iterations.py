import time,os,sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical
from deepModels import *

ROOTDIR = '../'
image_dir = ROOTDIR + '/data/2015/'

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
trainfile = ROOTDIR + '/data/2015-checked-train.txt'

np.random.seed(2017)
nx = 512
ny = 512

fcnmodel = getVgg16SegModel(ny,nx)
num_classes=2

fcnmodel.load_weights('vgg16-header_classifier.h5')
df = pd.read_csv('2015-checked-train.txt', header=None)
df2 = pd.read_csv('2015-Z-LOCATIONS.csv')
not_wildebeest = pd.DataFrame(columns=['image_name', 'xcoord', 'ycoord'])
not_wildebeest.to_csv(ROOTDIR + '/data/new_not_wildebeest_locations.csv', header=True, mode='w')

for i in range(df.size):
	x_pos, y_pos = 0,0
	filename = df.iloc[i][0]
	print('Examining image ' + str(i+1) + ' of ' + str(df.size) +  ': ' + filename)
	img = Image.open(image_dir + str(filename) + '.JPG')
	df3 = df2.loc[df2.image_name == filename]
	# imarr = np.zeros((img.size[0], img.size[1], 3))
	filenames = []
	x_ts = []
	y_ts = []
	tinyim_i = 0
	for y_pos in range(0, img.size[1], ny):
		for x_pos in range(0, img.size[0], nx):
			arr = np.array(img)
			arr = arr[x_pos:nx+x_pos,y_pos:ny+y_pos,:]
			arrx, arry = arr.shape[0], arr.shape[1]
			if arrx < nx:
				arr2 = np.zeros((nx - arrx, arry, 3))
				arr = np.concatenate((arr, arr2), axis=0)
			if arry < ny:
				arr2 = np.zeros((arr.shape[0], ny - arry, 3))
				arr = np.concatenate((arr, arr2), axis=1)
			arr = arr.astype('float32')/255.0
			X = []
			X.append(arr)
			X = np.asarray(X)
			X = X.astype('float32')/255
			result = fcnmodel.predict(X)[0]
			predicted_class = np.argmax(result, axis=2)
			for y_in in range(0, ny, 32):
				for x_in in range(0, nx, 32):
					flag = False
					if np.max(predicted_class[y_in:y_in+32,x_in:x_in+32]) > 0.95:
						x_t, y_t = x_pos + x_in + 15, y_pos + y_in + 15
						for index, d in df3.iterrows():
							if ((d.xcoord - x_t) ** 2) + ((d.ycoord - y_t) ** 2) <= 1600:
								flag = True
								break
						if flag == False:
							filenames.append(filename)
							x_ts.append(x_t)
							y_ts.append(y_t)
							if x_in >= 50 and y_in >= 50 and x_in <= nx - 50 and y_in <= ny - 50:
								tiny_im = arr[x_in - 32:x_in+32, y_in - 32, y_in+32, :]
								tiny_img = Image.fromarray(np.uint8(tiny_im))
								tiny_img.save(image_dir + 'test/' + filename + '_' + str(tiny_im_i) + '.png')
								tiny_im_i = tiny_im_i + 1
							# not_wildebeest = not_wildebeest.append({'image_name':filename, 'xcoord':x_t, 'ycoord':y_t}, ignore_index=True)
			# predicted_class = predicated_class * 255.0
			# predicted_class = np.concatenate((predicted_class, predicted_class, predicted_class) axis=2)
			# imarr[x_pos:x_pos+arrx,y_pos:arry+y_pos),:] = predicted_class[0:arrx,0:arry, :]
	# img = Image.fromarray(np.uint8(im_arr))
	# img.save('2015/test/' + str(filename) + '.png')
	not_wildebeest = pd.DataFrame(np.column_stack([filenames,x_ts,y_ts]), columns=['image_name', 'xcoord', 'ycoord'])
	not_wildebeest.to_csv(ROOTDIR + 'test/new_not_wildebeest_locations.csv', header=False, mode='a')

print('Finished.')
