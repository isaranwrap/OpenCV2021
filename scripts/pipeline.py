import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Read image here
#~cv2.imread()

# Initialize variables
imageFolder = '/Users/Praveens/Desktop/ishan/OpenCV2021/data/data/clean/cam2/output'
resizeDim = 160
num_channels = 3
num_filters = 8
filter_size = 3
pool_size = 2

# Mapping between food labels
food_to_label = dict()
label_to_food = dict()
for indx, food in enumerate(foods):
  food_to_label[food] = indx
  label_to_food[indx] = food

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model
# Read in classifiers
foodClassifier = create_model()
volumeClassifier = create_model()

labelLookUP = foodClassifier.predict(input_image) # rgb image
scalingFactor = volumeClassifier.predict(input_image) # left/depth/mask


# predict future
label = foodClassifier.predict()

# Label lookup table
lookup_table = pd.read_csv('lookup.csv')


#send string to app to print

## Take input image from OAK-D
import sys
import cv2
from pathlib import Path

parent = Path(__file__).parent.absolute()
sys.path.append(parent)

from mrcnn.config import Config

numDetectedFoods = 0
detectedFoods = []
logDir = //
mRCNNWeights = //

class DetectedObject:
    def __init__(self, food_name, vol):
        self.name = food_name
        self.vol = vol

    def setVol(self, vol):
        self.vol = vol


class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False
    NAME = 'mask'


#Instantiate pipeline
def startPipeline(rgbImage, depthImage, leftImage):
    #rgbImage, depthImage, and leftImage as paths
    inference_config = InferenceConfig()

    #loading in mrcnn
    mrcnn = MaskRCNN(mode="inference", 
                        config=inference_config,
                        model_dir=logDir)
    mrcnn.load_weights(mRCNNWeights, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    #run mrcnn on rgbImage
    results = mrcnn.detect([rgbImage], verbose=1)

    #handle network returns
    r = results[0]
    '''
    for i in r["masks"]:
        cv2.rectangle
    '''
    detectedFoods = len(r["masks"]) #verify this

    for i in range(detectedFoods):
        id = 'food'+str(i)#help from ish!
        id = DetectedObject(r["class_ids"][i],r["masks"][i])
        detectedFoods += id
        M = cv2.moments(r["masks"][i])
        # calculate centroid of mask
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # resize left, depth, and mask to resizeDimxresizeDim
        iX = cX - int(resizeDim/2)
        iY = cY - int(resizeDim/2)
        eX = cX + int(resizeDim/2)
        eY = cY + int(resizeDim/2)

        #cropRGB = rgbImage[iY:eY, iX:eX]

#csv.to_dict()
[food_to_calorie[food] for food in calorieLUTable]

calorie

