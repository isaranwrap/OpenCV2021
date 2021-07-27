import numpy as np
import pandas as pd
import os 

baseFolder = '/Users/saranmedical-smile/Desktop/opencv-2021/'

data = pd.read_csv(os.path.join(baseFolder, 'food-data/FoundationFoods/sample_food.csv'))


csvFolder = os.path.join(baseFolder, 'food-data/FoundationFoods/')

for file in os.listdir(os.path.join(baseFolder, 'food-data/FoundationFoods/')):
    if file[-4:] == '.csv':
        print(file[:-4], pd.read_csv(os.path.join(csvFolder, file)).shape)
        #print(pd.read_csv(os.path.join(baseFolder, file)).shape)
    #if file[:-4]
    #pd.read_csv(os.path.join(baseFolder, file))
#print(os.listdir(baseFolder))


tmp = pd.read_csv('/Users/saranmedical-smile/Desktop/opencv-2021/food-data/FoundationFoods/food_nutrient.csv')
