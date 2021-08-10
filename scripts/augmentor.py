import os
import cv2
import Augmentor

imagePath = '/Users/Praveens/Desktop/ishan/OpenCV2021/AugmentData'

def augmentFood(food, N = 1000):
    foodPath = '/Users/Praveens/Desktop/ishan/OpenCV2021/AugmentData/{}'.format(food)
    p = Augmentor.Pipeline(foodPath)
    p.zoom(probability = 0.8, min_factor = 0.5, max_factor = 1.5)
    p.flip_top_bottom(probability = 0.5)
    p.flip_left_right(probability = 0.5)
    p.rotate(max_left_rotation=15,max_right_rotation=15,probability=0.3)
    p.random_brightness(probability = 0.5, min_factor = 1.2, max_factor = 1.2)
    p.sample(N)

augmentFood('bagel',8)
augmentFood('cookie',15)
augmentFood('kiwi',16)
augmentFood('oj',12)
augmentFood('steak',10)
#p.random_distortion(probability = 0.5, grid_width = 3, grid_height = 3, magnitude = 1)
#p.sample(100)