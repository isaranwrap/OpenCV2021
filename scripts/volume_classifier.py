# %%
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af

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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, BatchNormalization


def one_hot(Y):
    oh = np.zeros((len(Y), Y.max() + 1))
    oh[np.arange(Y.size), Y] = 1
    oh = oh.T
    return oh

# Initialize variables
imageFolder = '/Users/Praveens/Desktop/ishan/OpenCV2021/data/data/clean/cam2/output'
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

height = np.array([0.5, 1.0, 1.4, 0.5, 0.9, 1.0]) # Height of water displacement (cm)
weight = np.array([4.5, 2.25, 4]) # Weight of food (in ounces)
volume = np.pi*2.2**2*height
#array([ 7.60265422, 15.20530844, 21.28743182,  7.60265422, 13.6847776 , 15.20530844])
<<<<<<< HEAD
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af
=======
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af

# Mapping between food labels
food_to_label = dict()
label_to_food = dict()
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af
food_to_vol = dict()

for indx, food in enumerate(foods):
  food_to_label[food] = indx
  label_to_food[indx] = food
  food_to_vol[food] = volume[indx]

# Read in raw images
#rawImages = np.array([cv2.imread(os.path.join(imageFolder, file)) for file in os.listdir(imageFolder)]) # 160 x 160 
rawLabels = np.array([re.compile(r'(?<=cam2_original_)[a-zA-Z]+').findall(s)[0] for s in os.listdir(imageFolder)])#change more so re.compile gets volume from img and not 'banana'
volumeLabels = np.array([food_to_vol[label] for label in rawLabels])

# Resize Z
trainX = np.array([cv2.resize(image, (resizeDim,resizeDim))/255. for image in rawImages])
#trainY = one_hot(rawLabels) 
#trainY = tf.keras.utils.to_categorical(rawLabels) # Equivalent to one_hot(rawLabels).T
scaler = MinMaxScaler(feature_range=(0,1))
trainY = scaler.fit_transform(volumeLabels.reshape(-1, 1))
multiplier = scaler.scale_[0] #i think [0] should correlate to volume label?
shift = scaler.min_[0]

model = Sequential()

model.add(Conv2D(128,(3,3), activation='relu', input_shape = trainX.shape[1:]))
model.add(MaxPool2D(2,2))
model.add(Dropout(.5))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(.5))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam", metrics = ['mse', 'mae', 'mape'])

model.fit(trainX, trainY, batch_size=256, epochs=20, validation_split=0.3)

model.summary()

<<<<<<< HEAD
#model predicts that volume of X is (model.predict(X)-shift)*multiplier
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af
=======
#model predicts that volume of X is (model.predict(X)-shift)*multiplier
>>>>>>> de57a1926e889fb89bd119da44f5c18e670530af
