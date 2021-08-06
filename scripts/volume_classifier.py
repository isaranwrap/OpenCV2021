# %%
# Imports
import os
import re
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten
import Augmentor # 
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, BatchNormalization

def one_hot(Y):
  oh = np.zeros((len(Y), Y.max()+1))
  oh[np.arange(Y.size), Y] = 1
  oh = oh.T
  return oh

# Initialize variables
parent = Path(__file__).parent.absolute()
imageFolder = str((Path(parent).parent.absolute() / Path('data/clean')).resolve().absolute()) #take out distortions
resizeDim = 32

foods = ['rgb','banana', 'potato', 'kiwi', 'egg','potato']
num_foods = len(foods)

# Mapping between food labels
food_to_label = dict()
label_to_food = dict()
for indx, food in enumerate(foods):
  food_to_label[food] = indx
  label_to_food[indx] = food

# Read in raw images
rawImages = np.array([cv2.imread(os.path.join(imageFolder, file)) for file in os.listdir(imageFolder)])
rawLabels = np.array([food_to_label[re.compile]])

# Define the neural network
model = tf.keras.Sequential([
                             Flatten(),
                             Dense(int(num_foods*4), activation='relu', # 144 neurons in HL1 
                                   input_shape=(784,),name='hidden1'), # 784 x 1
                             Dense(int(num_foods*2), activation='relu', name='hidden2'), # 72 neurons in HL2
                             Dense(num_foods, name='output_layer')])
model.compile(loss='binary_crossentropy',
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"),
metrics=['accuracy'])

rawTrainImages = [cv2.imread(os.path.join(imageFolder, file)) for file in os.listdir(imageFolder)]
rawLabels = np.array([food_to_label[food[:-5]] for food in os.listdir(imageFolder)])
labels = one_hot(rawLabels)