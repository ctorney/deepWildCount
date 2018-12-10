
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, sys, cv2
import time
from generator import BatchGenerator
sys.path.append("..")

from models.yolo_models import get_yolo


FINE_TUNE=1


LABELS = ['wildebeest']
IMAGE_H, IMAGE_W = 864, 864
GRID_H,  GRID_W  = 27, 27
# each cell is going to be 32x32
BOX              = 3
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

ANCHORS          = [53.57159857, 42.28639429, 29.47927551, 51.27168234, 37.15496912, 26.17125211]
ignore_thresh=0.8
NO_OBJECT_SCALE  = 2.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

if FINE_TUNE:
    BATCH_SIZE       = 4
else:
    BATCH_SIZE       = 16




train_image_folder = 'train_images/' 
train_annot_folder = 'train_images/'

model = get_yolo(IMAGE_W,IMAGE_H)

if FINE_TUNE:
    model.load_weights('../weights/wb-yolo.h5')
else:
    model.load_weights('../weights/yolo-v3-coco.h5', by_name=True)

    for layer in model.layers[:-7]:
        layer.trainable = False

print(model.summary())

def yolo_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
    y_true = tf.reshape(y_true, tf.concat([tf.shape(y_true)[:3], tf.constant([3, -1])], axis=0))

    # compute grid factor and net factor
    grid_h      = tf.shape(y_true)[1]
    grid_w      = tf.shape(y_true)[2]
    

    # the variable to keep track of number of batches processed
    batch_seen = tf.Variable(0.)        

    grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

    net_h       = IMAGE_H
    net_w       = IMAGE_W
    net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
    
    """
    Adjust prediction
    """

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [BATCH_SIZE, 1, 1, 3, 1])
    pred_box_xy    = (cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy


    pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
    pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
    pred_box_class = tf.sigmoid(y_pred[..., 5:])                                            # adjust class probabilities      
    # initialize the masks
    object_mask     = tf.expand_dims(y_true[..., 4], 4)

    """
    Adjust ground truth
    """
    true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
    true_box_wh    = y_true[..., 2:4] # t_wh
    true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
    true_box_class = y_true[..., 5:]         

    anc = tf.constant(ANCHORS, dtype='float', shape=[1,1,1,3,2])
    true_xy = tf.expand_dims(true_box_xy / grid_factor,4)
    true_wh = tf.expand_dims(tf.exp(true_box_wh) * anc / net_factor,4)
    """
    Compare each predicted box to all true boxes
    """        
    # initially, drag all objectness of all boxes to 0
    conf_delta  = pred_box_conf 

    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
    pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * anc / net_factor, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious   = tf.reduce_max(iou_scores, axis=-1)        
    conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)
    """
    Warm-up training
    """
    batch_seen = tf.assign_add(batch_seen, 1.)
    
    xywh_mask=object_mask
    """
    Compare each true box to all anchor boxes
    """      
    xywh_scale = tf.exp(true_box_wh) * anc / net_factor
    xywh_scale = tf.expand_dims(2 - xywh_scale[..., 0] * xywh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

    xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * xywh_scale
    wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * xywh_scale
    obj_delta  = (object_mask * (pred_box_conf-true_box_conf) * OBJECT_SCALE) 
    no_obj_delta = ((1-object_mask) * conf_delta) * NO_OBJECT_SCALE
    class_delta = object_mask * (pred_box_class-true_box_class)

    loss = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5))) + \
           tf.reduce_sum(tf.square(wh_delta),       list(range(1,5))) + \
           tf.reduce_sum(tf.square(obj_delta),     list(range(1,5))) + \
           tf.reduce_sum(tf.square(no_obj_delta),     list(range(1,5))) + \
           tf.reduce_sum(tf.square(class_delta),    list(range(1,5)))
    return loss



from operator import itemgetter
import random

### read saved pickle of parsed annotations
with open ('train_images/annotations-checked-2.pickle', 'rb') as fp:
    all_imgs = pickle.load(fp)

num_ims = len(all_imgs)
indexes = np.arange(num_ims)
random.shuffle(indexes)


train_imgs = list(itemgetter(*indexes[:].tolist())(all_imgs))

def normalize(image):
    image = image / 255.
    
    return image

train_batch = BatchGenerator(
        instances           = train_imgs, 
        anchors             = ANCHORS,   
        labels              = LABELS,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 1000,
        batch_size          = BATCH_SIZE,
        min_net_size        = IMAGE_H,
        max_net_size        = IMAGE_H,   
        shuffle             = False, 
        jitter              = 0.0, 
        norm                = normalize
)


if FINE_TUNE:
    optimizer = Adam(lr=0.5e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    EPOCHS=20
else:
    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    EPOCHS=25

model.compile(loss=yolo_loss, optimizer=optimizer)
wt_file='../weights/wb-yolo-aug.h5'

early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=5, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint(wt_file, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


start = time.time()
model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = EPOCHS, 
                    verbose          = 1,
                    callbacks        = [checkpoint, early_stop],
                    max_queue_size   = 3)
end = time.time()
print('Training took ' + str(end - start) + ' seconds')
model.save_weights(wt_file)

