from __future__ import print_function
import keras
from keras.models import Sequential	
from keras.layers import *
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa
import cv2
from keras.optimizers import SGD
from keras.constraints import maxnorm

pickle_file = "C:\\deepLearning\\data_batch_1"
with open(pickle_file,'rb') as f:
    save = pickle.load(f, encoding = 'bytes')
    train_dataset = np.array(save[b'data'],dtype=np.uint8)
    train_labels = np.array(save[b'labels'])


for i in range(4):
    pickle_file = "C:\\deepLearning\\data_batch_"+str(i+2)
    print(pickle_file)
    with open(pickle_file,'rb') as f:
        save = pickle.load(f, encoding = 'bytes')
        train_dataset_ap = np.array(save[b'data'],dtype=np.uint8)
        train_labels_ap = np.array(save[b'labels'])
        train_dataset = np.append(train_dataset, train_dataset_ap, axis = 0)
        train_labels = np.append(train_labels, train_labels_ap, axis = 0)

pickle_file = "C:\\deepLearning\\test_batch"
with open(pickle_file,'rb') as f:
    save = pickle.load(f, encoding = 'bytes')
    test_dataset = np.array(save[b'data'],dtype=np.uint8)
    test_labels = np.array(save[b'labels'])

image_size = 32
channels = 3
num_labels = 10

def reshape(data, labels):
    data = (np.array(data.reshape(-1, channels, image_size, image_size))/255.0)
    data = data.transpose([0,2,3,1]).astype(np.float32)
    labels = (np.arange(num_labels)==labels[:,None]).astype(np.float32)
    return data, labels


train_dataset, train_labels = reshape(train_dataset, train_labels)
test_dataset, test_labels = reshape(test_dataset, test_labels)

seq = iaa.Sequential([
   iaa.Fliplr(1)
])


images = train_dataset
images_aug = seq.augment_images(images)
train_dataset = np.append(train_dataset, images_aug, axis = 0)
train_labels = np.append(train_labels, train_labels, axis = 0)

model = Sequential()
model.add(Conv2D(32,(3,3), activation = 'relu',input_shape=(image_size, image_size, channels), padding = "same"))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

epochs = 25
lrate = 0.01
decay = lrate/epochs
opt = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

model.fit(train_dataset, train_labels, batch_size=64, epochs=epochs, shuffle="true")
score = model.evaluate(test_dataset, test_labels)
print(model.summary())
print('Test accuracy:', score[1])
