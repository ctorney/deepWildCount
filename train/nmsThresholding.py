
import numpy as np
import pickle
import os, sys, cv2
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

def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

### read saved pickle of parsed annotations
with open ('train_images/annotations-checked-2.pickle', 'rb') as fp:
    all_imgs = pickle.load(fp)

nmsList = []
for im in all_imgs:
    all_objs = im['object']
    boxes=[]
            
    for obj in all_objs:

       box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]    
       boxes.append(box)

           
                

    for i in range(len(boxes)):

        for j in range(i+1, len(boxes)):
            iou = bbox_iou(boxes[i][0:4], boxes[j][0:4])
            if iou>0:
                nmsList.append(iou)

for nms in nmsList:
    print(nms)
