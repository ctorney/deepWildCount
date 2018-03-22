
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
#import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, sys, cv2
import time
from preprocessing import parse_annotation, BatchGenerator
sys.path.append("..")
from yolo_models import get_yolo2


FROM_VOC=1
CROSS_ENT=0

# In[93]:


LABELS = ['wildebeest']

IMAGE_H, IMAGE_W = 416, 416
# predictions reduce resolution by x32
GRID_H,  GRID_W  = 13 , 13
# each cell is going to be 32x32
BOX              = 1
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5 # this must be for showing the object - should be lower??
NMS_THRESHOLD    = 0.3#0.45 # non max suppression - what does this do?

# this is the width/height of the anchor boxes - this will be 2,2 for all 5 - maybe - might be better to use pretrained
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = [2.0,2.0]

# scales - for training maybe?? no idea
# all seem to be in the custom loss function - some method to weight the loss
NO_OBJECT_SCALE  = 1.0# upping this to 5 (from 1) to get rid of false positives
OBJECT_SCALE     = 5.0
COORD_SCALE      = 2.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 16
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50
print(len(LABELS))


#true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))


train_image_folder = 'train_images/' #/home/ctorney/data/coco/train2014/'
train_annot_folder = 'train_images/'
valid_image_folder = train_image_folder#'/home/ctorney/data/coco/val2014/'
valid_annot_folder = train_annot_folder#'/home/ctorney/data/coco/val2014ann/'


model = get_yolo2(IMAGE_W,IMAGE_H)


# In[97]:


#freeze darknet-19
# each grid cell is going to predict 80 - classes + 5 bounding box parameters, 2*size/pos and one for objectedness??


# # Load pretrained weights

# **Load the weights originally provided by YOLO**

# In[98]:


# transfer learning from coco
if FROM_VOC:
    model.load_weights('../weights/weights_coco.h5', by_name=True)
    #model.load_weights('../weights/wb_ce_yolo.h5') 
#model.load_weights('../weights/darknet19-cls.h5', by_name=True)
#weight_reader = WeightReader(wt_path)

#    layer   = model.layers[-3] # the last convolutional layer
#    weights = layer.get_weights()

#    new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
#    new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

 #   layer.set_weights([new_kernel, new_bias])
    layer   = model.layers[-2] # the last convolutional layer
    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
    new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

    layer.set_weights([new_kernel, new_bias])
#    for layer in model.layers[:-3]:
#        layer.trainable = False
else:
    model.load_weights('../weights/wb_yolo.h5') 
print(model.summary())

def custom_loss_old(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, BOX, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    ### adjust w and h
    pred_box_wh = tf.ones(tf.shape(y_pred[..., 0:2]))  * np.reshape(ANCHORS, [1,1,1,BOX,2])
 #   pred_box_wh = tf.exp(0.*y_pred[..., 2:4])
    #pred_box_wh = np.reshape(ANCHORS, [1,1,1,BOX,2])
    
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
   # pred_box_class = y_pred[..., 5:]
    
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
    ### adjust w and h
    true_box_wh = pred_box_wh #y_true[..., 2:4] # number of cells accross, horizontally and vertically
    #true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    
    true_box_conf = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
   # true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    
    ### confidence mask: penalize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    #true_xy = true_boxes[..., 0:2]
    #true_wh = true_boxes[..., 2:4]
    
    #true_wh_half = true_wh / 2.
    #true_mins    = true_xy - true_wh_half
    #true_maxes   = true_xy + true_wh_half
    
    #pred_xy = tf.expand_dims(pred_box_xy, 4)
    #pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    #pred_wh_half = pred_wh / 2.
    #pred_mins    = pred_xy - pred_wh_half
    #pred_maxes   = pred_xy + pred_wh_half    
    
    #intersect_mins  = tf.maximum(pred_mins,  true_mins)
    #intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    #intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    #intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    #true_areas = true_wh[..., 0] * true_wh[..., 1]
    #pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    #union_areas = pred_areas + true_areas - intersect_areas
    #iou_scores  = tf.truediv(intersect_areas, union_areas)

    #best_ious3 = tf.reduce_max(iou_scores, axis=4)
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=-1)
    best_ious = tf.expand_dims(best_ious, -1)
    ## CHANGED UP TO HERE
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
 #   class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
  #  nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    #loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    #loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    #loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    

    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    true_box_conf = y_true[..., 4]
    loss_fp  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf))
 #   loss = loss/2. + loss_fp/2.
 #   loss = loss_xy + loss_wh + loss_conf #+ loss_class
    loss = loss_xy + loss_conf 
    
    #nb_true_box = tf.reduce_sum(y_true[..., 4])
    #nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """    
    #current_recall = nb_pred_box/(nb_true_box + 1e-6)
    #total_recall = tf.assign_add(total_recall, current_recall) 

    #loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    #loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    #loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    #loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    #loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    #loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    #loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    #loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(best_ious)], message='pred \t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(best_ious3)], message='true \t', summarize=1000)
    
    return loss

def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:3]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W,1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1,1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.ones(mask_shape)
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    ### adjust confidence
    pred_box_conf = y_pred[..., 2]
    
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 2], axis=-1) * COORD_SCALE
    
    
    true_box_conf = y_true[..., 2]
    
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.


 #   loss_fp  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf))
 #   loss = loss/2. + loss_fp/2.
    loss = loss_xy + loss_conf#+ loss_conf + loss_class

    
    return loss
    
    #nb_true_box = tf.reduce_sum(y_true[..., 4])
    #nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """    
    #current_recall = nb_pred_box/(nb_true_box + 1e-6)
    #total_recall = tf.assign_add(total_recall, current_recall) 

    #loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    #loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    #loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    #loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    #loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    #loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    #loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    #loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(best_ious)], message='pred \t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(best_ious3)], message='true \t', summarize=1000)
    
    return loss


def custom_ce_loss(y_true, y_pred):
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    true_box_conf = y_true[..., 4]
 #   cce = binary_crossentropy(true_box_conf, pred_box_conf) 
 #   cce = tf.Print(cce, [tf.shape(true_box_conf)], message='true \t', summarize=1000)
 #   cce = tf.Print(cce, [tf.shape(pred_box_conf)], message='pred \t', summarize=1000)
    cce  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf))
    return cce
    

def focal_loss(y_true, y_pred):
    gamma=2.
    alpha=0.5
    y_pred_in = tf.sigmoid(y_pred[..., 4])
    y_true_in = y_true[..., 4]
    pt_1 = tf.where(tf.equal(y_true_in, 1), y_pred_in, tf.ones_like(y_pred_in))
    pt_0 = tf.where(tf.equal(y_true_in, 0), y_pred_in, tf.zeros_like(y_pred_in))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}



from operator import itemgetter
import random

### read saved pickle of parsed annotations
with open ('train_images/annotations.pickle', 'rb') as fp:
    all_imgs = pickle.load(fp)

num_ims = len(all_imgs)
indexes = np.arange(num_ims)
random.shuffle(indexes)

num_val = 0#num_ims//10

#valid_imgs = list(itemgetter(*indexes[:num_val].tolist())(all_imgs))
train_imgs = list(itemgetter(*indexes[num_val:].tolist())(all_imgs))

def normalize(image):
    image = image / 255.
    
    return image
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize, jitter=False)
#valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)


# In[104]:




# **Setup a few callbacks and start the training**

# In[105]:



if CROSS_ENT:
    optimizer = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
    #optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
 #   model.compile(loss=custom_ce_loss, optimizer=optimizer)
    model.compile(loss=focal_loss, optimizer=optimizer)

    wt_file='../weights/wb_ce_yolo.h5'

else:
    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  #  optimizer = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
    model.compile(loss=custom_loss_old, optimizer=optimizer)
    wt_file='../weights/wb_yolo.h5'
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=5, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint(wt_file, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


start = time.time()
model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 4, 
                    verbose          = 1,
            #        validation_data  = valid_batch,
            #        validation_steps = len(valid_batch),
 #                   callbacks        = [checkpoint],# , early_stop],#, tensorboard], 
                    max_queue_size   = 3)
end = time.time()
print('Training took ' + str(end - start) + ' seconds')
model.save_weights(wt_file)


# # Perform detection on image

# In[7]:


#model.load_weights("weights_coco.h5")

#dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))


# In[9]:

#
#image = cv2.imread('raccoon-190.jpg')
#
#plt.figure(figsize=(10,10))
#
#input_image = cv2.resize(image, (416, 416))
#input_image = input_image / 255.
#input_image = input_image[:,:,::-1]
#input_image = np.expand_dims(input_image, 0)
#
#netout = model.predict([input_image, dummy_array])
#
#boxes = decode_netout(netout[0], 
#                      obj_threshold=0.1,#OBJ_THRESHOLD,
#                      nms_threshold=NMS_THRESHOLD,
#                      anchors=ANCHORS, 
#                      nb_class=CLASS)
#image = draw_boxes(image, boxes, labels=LABELS)
#
#plt.imshow(image[:,:,::-1]); plt.show()
#
#
# # Perform detection on video

# In[8]:

#
#model.load_weights("weights_coco.h5")
#
#dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
#
#
## In[11]:
#
#
#video_inp = '../basic-yolo-keras/images/phnom_penh.mp4'
#video_out = '../basic-yolo-keras/images/phnom_penh_bbox.mp4'
#
#video_reader = cv2.VideoCapture(video_inp)
#
#nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
#frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#
#video_writer = cv2.VideoWriter(video_out,
#                               cv2.VideoWriter_fourcc(*'XVID'), 
#                               50.0, 
#                               (frame_w, frame_h))
#
#for i in tqdm(range(nb_frames)):
#    ret, image = video_reader.read()
#    
#    input_image = cv2.resize(image, (416, 416))
#    input_image = input_image / 255.
#    input_image = input_image[:,:,::-1]
#    input_image = np.expand_dims(input_image, 0)
#
#    netout = model.predict([input_image, dummy_array])
#
#    boxes = decode_netout(netout[0], 
#                          obj_threshold=0.3,
#                          nms_threshold=NMS_THRESHOLD,
#                          anchors=ANCHORS, 
#                          nb_class=CLASS)
#    image = draw_boxes(image, boxes, labels=LABELS)
#
#    video_writer.write(np.uint8(image))
#    
#video_reader.release()
#video_writer.release()  
#
