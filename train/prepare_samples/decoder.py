
import numpy as np
import pandas as pd
import os,sys
import cv2


MIN_AREA=32*16
MAX_AREA=64*64
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

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
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union



def decode_netout(netout, anchors, xvals, yvals, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
 #   netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]
            
            if(objectness <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) * net_w / grid_w # center position, unit: image width
            y = (row + y) * net_h / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) #/ net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) #/ net_h # unit: image height  

            if (w*h > MAX_AREA): continue
            if (h > 65): continue
            if (w > 65): continue
            if (w*h < MIN_AREA): continue
            if x-w/2<0: continue
            if y-h/2<0: continue
            if x+w/2>net_w: continue
            if y+h/2>net_h: continue

            w_in_box = np.sum( (xvals>(x-w/2)) & (yvals>(y-h/2)) & (xvals<(x+w/2)) & (yvals<(y+h/2)))
            if w_in_box>2: continue
            if w_in_box<1: continue
            #print(xvals)
            #print(yvals)
            #print(x-w/2, y-h/2, x+w/2, y+h/2)
            #print(w_in_box)


            
            # last elements are class probabilities
            classes = netout[row,col,b,5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes

        
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    sorted_indices = np.argsort([-box.objness for box in boxes])

    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]

        if boxes[index_i].objness == 0: continue

        for j in range(i+1, len(sorted_indices)):
            index_j = sorted_indices[j]

            if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                boxes[index_j].objness = 0
    return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def do_duplicates(boxes, xvals, yvals):

    nbox = len(boxes)
    xmins = np.asarray([b.xmin for b in boxes])
    xmaxs =  np.asarray([b.xmax for b in boxes])
    ymins =  np.asarray([b.ymin for b in boxes])
    ymaxs =  np.asarray([b.ymax for b in boxes])
    objness =  np.asarray([b.objness for b in boxes])

    for p in range(len(xvals)):
        in_boxes = (xvals[p]>xmins)&(xvals[p]<xmaxs)&(yvals[p]>ymins)&(yvals[p]<ymaxs)
        bx_count = np.sum(in_boxes)
        if bx_count==0:
            box = BoundBox(xvals[p]-32,yvals[p]-32,xvals[p]+32,yvals[p]+32, 1.0)

            boxes.append(box)
            continue


        if bx_count>1:
            this_obj = objness.copy()
            this_obj[np.logical_not(in_boxes)]=0
            in_boxes[np.argmax(this_obj)]=False
            inds = np.argwhere(in_boxes)
            for i in inds:
                boxes[i[0]].objness=0

def draw_boxes(image, boxes, xvals,yvals):
    for box in boxes:
        
 #       bx_area = (box.xmin-box.xmax)**2+(box.ymin-box.ymax)**2
 #       print(bx_area)
        #if ((box.xmax-box.xmin)*(box.ymax-box.ymin))>(64*64):
        #    continue
       # for i in range(len(labels)):
 #           if box.classes[i] > obj_thresh:
        #    label_str += labels[i]
                #print(label_str)
         #   label = i
                #print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                #print(box.xmin,box.ymin,box.xmax,box.ymax)
                
        #if label >= 0:
        if box.objness>0:
            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 2)
#    for i in range(len(xvals)):
#        cv2.rectangle(image,  (int(xvals[i])-32,int(yvals[i])-32), (int(xvals[i])+32,int(yvals[i])+32),  (0,0,255), 1)

            #cv2.putText(image, 
            #            label_str + ' ' + str(box.get_score()), 
            #            (box.xmin, box.ymin - 13), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 
            #            1e-3 * image.shape[0], 
            #            (0,255,0), 2)
        
    return image      
# set some parameters
obj_thresh, nms_thresh = 0.001, 0.45
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

def decode(yolos, im_size, xvals, yvals, save_name, img):
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], xvals, yvals, im_size, im_size)

# correct the sizes of the bounding boxes
  #  correct_yolo_boxes(boxes, im_size, im_size, im_size, im_size)

# suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    do_duplicates(boxes,xvals,yvals)
# draw bounding boxes on the image using labels
    draw_boxes(img, boxes,  xvals,yvals) 

# write the image with bounding boxes to file
    if len(boxes)>0:
        cv2.imwrite(save_name, (img).astype('uint8')) 

    return_boxes = []

    for b in boxes:
        if b.objness>0:
            return_boxes.append([b.xmin,b.ymin,b.xmax,b.ymax])

    return return_boxes
