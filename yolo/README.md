

** YOLO implementation of wildebeest count **

Step 1 - utils/conv_yolo_keras.py 
Convert pretrained darknet weights to keras

Step 2 - fine tune the darknet-19 classifier
train/class_train/train_head.py - train the final conv layer first
train/class_train/fine_tune.py - train all layers 

Step 3 - train yolo first pass
train/train.py

Step 4 - predict the test images and use false positives to train the classifier again
train/class_train/negatives_1.py
train/class_train/fine_tune.py
