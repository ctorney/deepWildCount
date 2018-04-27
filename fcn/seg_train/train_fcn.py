from data_generator import *
import numpy as np
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
from keras.losses import categorical_crossentropy
import keras.backend as K
from fcn_networks import *
import time
import tensorflow as tf

#loss function
def weighted_categorical_crossentropy_fcn_loss(y_true, y_pred):
    # y_true is a matrix of weight-hot vectors (like 1-hot, but they have weights instead of 1s)
    y_true_mask = K.sign(y_true)  # [0 0 W 0] -> [0 0 1 0] where W > 0.
    cce = categorical_crossentropy(y_true_mask, y_pred)  # one dim less (each 1hot vector -> float number)
    y_true_weights_maxed = K.max(y_true, axis=-1)  # [0 120 0 0] -> 120 - get weight for each weight-hot vector
    wcce = cce * y_true_weights_maxed
    return (wcce)

model_type = 8 # model can be 32s, 16s or 8s
batch_size = 1
batch_size_val = 1
epochs = 10
lr_base = 1e-7#0.01 * (float(batch_size) / 16)
target_size = (512, 512)
class_weights = [1,19000]#0.50038, 666.47713]
class_weights = [1,1]#0.50038, 666.47713]

ROOTDIR = '../../'

# all training data
train_file_path = os.path.expanduser(ROOTDIR + '/data/2015-checked-train.txt')
# images with wildebeest present only
train_file_path = os.path.expanduser(ROOTDIR + '/data/2015-checked-train-w.txt')
val_file_path   = os.path.expanduser(ROOTDIR + '/data/2015-checked-train.txt')
data_dir        = os.path.expanduser( ROOTDIR + '/data/2015/')
label_dir       =  os.path.expanduser( ROOTDIR + '/data/2015/labels/')
data_suffix     = '.JPG'
label_suffix    = '.png'
classes = 2

# ###################### loss function & metric ########################
loss_fn = weighted_categorical_crossentropy_fcn_loss
metrics = ['accuracy']

#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=config)
#K.set_session(session)


input_shape = target_size + (3,)
batch_shape = (batch_size,) + input_shape

###########################################################
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = './'#os.path.join(current_dir, model_name)

checkpoint_path = os.path.join(save_path, 'checkpoint_weights.h5')

if model_type==8:
    model = fcn_8s_model()# getSegModel(input_width=target_size[0], input_height=target_size[1])
if model_type==16:
    model = fcn_16s_model()# getSegModel(input_width=target_size[0], input_height=target_size[1])
if model_type==32:
    model = fcn_32s_model()# getSegModel(input_width=target_size[0], input_height=target_size[1])

# comment out to refresh model
# model.load_weights(checkpoint_path)
#model.load_weights('fcn_cifar10_weights_from_classifier.h5')

#optimizer = Nadam()
optimizer = SGD(lr=lr_base) #, momentum=0.9)

#model.compile(loss=loss_fn,
model.compile(loss=loss_fn,#'categorical_crossentropy',
              optimizer=optimizer,
              metrics=metrics)

print(model.summary())

# ################### checkpoint saver ######################
checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.h5'), save_weights_only=True)#.{epoch:d}
callbacks = [checkpoint]

# ################### early stopping ########################
earlystopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
#callbacks.append(earlystopping)

# set data generator and train

def get_file_len(file_path):
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    return len(lines)

# epoch should correspond to running through the entire training set once
steps_per_epoch = int(np.ceil(((7360 // target_size[0]) * (4912 // target_size[1]) * get_file_len(train_file_path)) / float(batch_size)))

training_generator = SegDataGen(train_file_path,target_size[0],target_size[1],batch_size, class_weights, data_dir, label_dir).generate(classes)
test_generator = SegDataGen(val_file_path,target_size[0],target_size[1],batch_size_val, class_weights, data_dir, label_dir).generate(classes)

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    callbacks = callbacks,
    validation_steps = 32,#steps_per_epoch // 5, #int(np.ceil(((7360 // target_size[0]) * (4912 // target_size[1]) * get_file_len(train_file_path)) / float(batch_size))), 
    validation_data = test_generator
    )

model.save_weights('fcn_final' + str(model_type) + 's_weights.h5')
