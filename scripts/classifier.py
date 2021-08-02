# Imports
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten
import Augmentor # 

# Initialize variables
imageFolder = '/Users/Praveens/Desktop/ishan/OpenCV2021/data/data/clean'

foods = ['banana', 'potato', 'kiwi', 'egg']
num_foods = len(foods)

# Mapping between food labels
food_to_label = dict()
label_to_food = dict()
for indx, food in enumerate(foods):
  food_to_label[food] = indx
  label_to_food[indx] = food

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

trainImages = [cv2.imread(os.path.join(imageFolder, file)) for file in os.listdir(imageFolder)]
trainLabel = []