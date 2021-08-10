# Imports
import re
import os
import cv2
import random
import Augmentor
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, BatchNormalization


def one_hot(Y):
    oh = np.zeros((len(Y), Y.max() + 1))
    oh[np.arange(Y.size), Y] = 1
    oh = oh.T
    return oh

# Initialize variables
imageFolder = '/Users/Praveens/Desktop/ishan/OpenCV2021/data/data/clean/cam2/output'
foodClassifierPath = '/Users/Praveens/Desktop/ishan/OpenCV2021/models/'
resizeDim = 32
num_channels = 3
num_filters = 8
filter_size = 3
pool_size = 2


'''
foods = ['rgb', 'banana', 'potato', 'kiwi', 'egg', 'tomato', 
         'squash', 'corn', 'lettuce', 'cabbage', 'green_tea'
         'burger','steak', 'spaghetti']
'''
foods = ['rgb', 'banana', 'potato', 'kiwi', 'egg', 'tomato']
num_foods = len(foods)
num_foods_ = 32 # Used in NN initialization

# Mapping between food labels
food_to_label = dict()
label_to_food = dict()
for indx, food in enumerate(foods):
  food_to_label[food] = indx
  label_to_food[indx] = food

# Read in raw images
rawImages = np.array([cv2.imread(os.path.join(imageFolder, file)) for file in os.listdir(imageFolder)]) # 160 x 160 
rawLabels = np.array([food_to_label[re.compile(r'(?<=cam2_original_)[a-zA-Z]+').findall(s)[0]] for s in os.listdir(imageFolder)])

# Resize Z
trainX = np.array([cv2.resize(image, (resizeDim,resizeDim))/255. for image in rawImages])
trainY = one_hot(rawLabels) 
trainY = tf.keras.utils.to_categorical(rawLabels) # Equivalent to one_hot(rawLabels).T

model = Sequential()

model.add(Conv2D(128,(3,3), activation='relu', input_shape = trainX.shape[1:]))
model.add(MaxPool2D(2,2))
model.add(Dropout(.5))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(.5))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(num_foods))

model.compile(loss="mean_squared_error",optimizer="adam", metrics = ['mse', 'mae', 'mape'])

history = model.fit(trainX, trainY, batch_size=256, epochs=20, validation_split=0.3)

model.summary()

##foodClassifier = create_model()
#foodClassifier.load_weights('/Users/Praveens/Desktop/ishan/OpenCV2021/models/')
