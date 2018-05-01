import numpy as np
import os, cv2, sys
import time
sys.path.append("..")
from models.yolo_models import get_yolo

ROOTDIR = '../../'
image_dir = ROOTDIR + '/data/2015/'
train_images = np.genfromtxt(ROOTDIR + '/data/2015-checked-test.txt',dtype='str')
correct = np.genfromtxt(ROOTDIR + '/data/2015-checked-test-correct.txt',dtype='int')
IMAGES = len(train_images)


anchors = np.array([[53.57159857, 42.28639429], [29.47927551, 51.27168234], [37.15496912, 26.17125211]])
obj_thresh=0.5
nms_thresh=0.25
nb_box=3

IMAGE_H, IMAGE_W = 4928, 7360
NET_H, NET_W = 4928//2, 7360//2

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

n=0
error = 0.0
pmerror = 0.0
rmserror = 0.0
model = get_yolo(NET_W,NET_H)
model.load_weights('../weights/wb-yolo.h5')

for filename in train_images: 



    image = cv2.imread(image_dir + str(filename) + '.JPG')
    input_image = cv2.resize(image, (IMAGE_W, IMAGE_H))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    im_count = 0
    for xc in range(2):
        for yc in range(2):

            pred_image = input_image[yc*NET_H:(yc+1)*NET_H,xc*NET_W:(xc+1)*NET_W,:]
            pred_image = np.expand_dims(pred_image, 0)
            netout = model.predict(pred_image)[0]

            grid_h, grid_w = netout.shape[:2]
            netout = netout.reshape(grid_h,grid_w,nb_box,-1)

            # convert from raw output
            netout[..., :2]  = _sigmoid(netout[..., :2])
            netout[..., 4:]  = _sigmoid(netout[..., 4:])
            netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]

            # process the coordinates
            x = np.linspace(0, grid_w-1, grid_w)
            y = np.linspace(0, grid_h-1, grid_h)

            xv,yv = np.meshgrid(x, y)
            xv = np.expand_dims(xv, -1)
            yv = np.expand_dims(yv, -1)
            xpos =(np.tile(xv, (1,1,3))+netout[...,0]) * IMAGE_W / grid_w 
            ypos =(np.tile(yv, (1,1,3))+netout[...,1]) * IMAGE_H / grid_h
            wpos = np.exp(netout[...,2])
            hpos = np.exp(netout[...,3])


            for b in range(nb_box):
                wpos[...,b] *= anchors[b,0]
                hpos[...,b] *= anchors[b,1]

            objectness = netout[...,5]

            # select only objects above threshold
            indexes = objectness > obj_thresh


            new_boxes = np.column_stack((xpos[indexes]-wpos[indexes]/2, \
                                         ypos[indexes]-hpos[indexes]/2, \
                                         xpos[indexes]+wpos[indexes]/2, \
                                         ypos[indexes]+hpos[indexes]/2, \
                                         objectness[indexes]))

            # do nms 
            sorted_indices = np.argsort(-new_boxes[:,4])
            boxes=new_boxes.tolist()

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if new_boxes[index_i,4] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= nms_thresh:
                        new_boxes[index_j,4] = 0

            new_boxes = new_boxes[new_boxes[:,4]>0]
            im_count+=len(new_boxes)

    error += abs(im_count-correct[n])
    rmserror += (im_count-correct[n])**2
    pmerror += (im_count-correct[n])
    print(filename + ": " +  str(im_count) + " " + str(correct[n]) + " " + str(error/float(n+1)) + ", overcount " + str(pmerror/float(n+1))+ ", rms error " + str((rmserror/float(n+1))**0.5))

    n=n+1

#IMAGE_H, IMAGE_W = 4928, 7360




