import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, BatchNormalization

# https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = np.asarray(data)
data = data.reshape(1, 8, 8, 1)
# data shape is 1 x 8 x 8 x 1 

# Create initial weight matrix
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [np.asarray(detector), np.asarray([0.0])]

# Initialize model
model = keras.models.Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
model.build(input_shape = (1, 8, 8, 1))


# Set weights in model
model.set_weights(weights)

# apply filter to input data
yhat = model.predict(data)

# See yhat
print(yhat.reshape(6,6))
model.summary()