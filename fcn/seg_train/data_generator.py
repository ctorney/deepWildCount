import numpy as np
import pandas as pd
import glob
import itertools
import random
import cv2


class SegDataGen(object):

    def __init__(self, file_path = None, dim_x = 2000, dim_y = 2000, batch_size = 16, class_weight = None, images_path =None, segs_path=None):
        self.file_path = file_path
        self.dim_x = int(dim_x)
        self.dim_y = int(dim_y)
        self.batch_size = int(batch_size)
        self.class_weight = class_weight

        assert images_path[-1] == '/'
        assert segs_path[-1] == '/'
        if self.file_path:
            df = pd.read_csv(self.file_path, header=None)
            df = images_path + df
            df = df + '.JPG'

        images = glob.glob( images_path + "*.JPG"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
        images.sort()
        segmentations  = glob.glob( segs_path + "*.JPG"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
        segmentations.sort()

        assert len( images ) == len(segmentations)
        for im , seg in zip(images,segmentations):
            assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

        if self.file_path:
            for im in images:
                if not im in list(df.values.flatten()):
                    del segmentations[images.index(im)]
                    del images[images.index(im)]
          #      else: 
          #          filename = segmentations[images.index(im)]
          #          img = cv2.imread(filename)
          #          img = np.array(img)[:, : , 0]
          #          img = np.around(img  / 255.0)

          #          if np.sum(img)<1e-6:
          #              del segmentations[images.index(im)]
          #              del images[images.index(im)]
#        print(len(images))

        zipped = zip(images,segmentations)
        im = ''
        seg = ''
        im_y, im_x = cv2.imread(df.iloc[0].values[0]).shape[0:2]
        x_range = im_x // self.dim_x
        y_range = im_y // self.dim_y
        prod = list(itertools.product(list(zipped), list(range(x_range)), list(range(y_range))))
        random.shuffle(prod)
        self.zipped = itertools.cycle(prod)


    def getImageArr(self, filename, x_pos, y_pos):
        width = self.dim_x
        height = self.dim_y
        left = x_pos * self.dim_x
        top = y_pos * self.dim_y

        try:
            img = cv2.imread(filename)
            img = img[top:top+height,left:left+width,:]
            img = np.array(img).astype(np.float32)
            img = img/255.0
            assert img.shape == (width, height, 3)

            return img

        except Exception as e:
            print(e)
            img = np.zeros((  width , height  , 3 ))
            return img


    def getSegmentationArr(self, filename , nClasses, x_pos, y_pos):
        width = self.dim_x
        height = self.dim_y
        left = x_pos * self.dim_x
        top = y_pos * self.dim_y

        seg_labels = np.zeros((  width , height  , nClasses ))
        try:
            img = cv2.imread(filename)
            img = img[top:top+height,left:left+width,:]
            img = np.array(img)[:, : , 0]
            #print(np.sum(img[:,:]))
            #correct for pixel intensity
            img = np.around((img * (nClasses - 1)) / 255.0)

            for c in range(nClasses):
                seg_labels[: , : , c ] = ((img == c ).astype(int)* (1 if self.class_weight is None else self.class_weight[c]))

        except Exception as e:
            print (e)
        
        seg_labels = np.reshape(seg_labels, ( width, height , nClasses ))
            
        return seg_labels


    def generate( self, n_classes=2):
    
        while True:
            X = []
            Y = []
            i = 0
            while i<self.batch_size:
                (im , seg), x_pos, y_pos = self.zipped.__next__()
                #print(im,seg,str(x_pos),str(y_pos))
                labels = self.getSegmentationArr(str(seg), n_classes, x_pos, y_pos)
  #              if (np.sum(labels[:,:,1])>0) or (random.uniform(0,1)<0.01):
                i = i + 1
                X.append(self.getImageArr(str(im), x_pos, y_pos))
                Y.append(labels)#self.getSegmentationArr(str(seg), n_classes, x_pos, y_pos))

            X = np.array(X)
            Y = np.array(Y)
            yield X, Y
