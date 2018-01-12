
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential, Model
import numpy as np

num_classes = 1000

model = VGG19(weights='imagenet',include_top=True)


base_model = VGG19(weights='imagenet',include_top=False,input_shape=(64,64,3) )


# CREATE A TOP MODEL
#model3 = Sequential()
#model3.add(Flatten(


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)

for layer in fcn_model.layers:
    layer.trainable = False

fcn_model.add(Conv2D(4096, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2'))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='predictions'))
fcn_model.add(Flatten())

# CONCATENATE THE TWO MODELS
#new_model.add(model3)
print(model.summary())
print(fcn_model.summary())

fcn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


training_images = []
training_labels = []

for subdir, dirs, files in os.walk('/data/datasets/imagenet_resized/train/'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                training_images.append(os.path.join(folder_subdir, file))
                training_labels.append(label_counter)

        label_counter = label_counter + 1

print(len(training_images))
print(len(training_labels))
import random

perm = list(range(len(training_images)))
random.shuffle(perm)
training_images = [training_images[index] for index in perm]
training_labels = [training_labels[index] for index in perm]

#for layer in model.layers:
#    weights = layer.get_weights()
#    for i in range(len(weights)):
#        print(layer.name, ' ', i, ': ', weights[i].shape)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = fcn_model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357),
