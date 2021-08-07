import os
import cv2
import Augmentor

imagePath = '/Users/Praveens/Desktop/ishan/OpenCV2021/data/data/clean/cam2'
p = Augmentor.Pipeline(imagePath)
p.zoom(probability = 0.8, min_factor = 0.5, max_factor = 1.5)
p.flip_top_bottom(probability = 0.5)
p.flip_left_right(probability = 0.5)
p.random_brightness(probability = 0.5, min_factor = 1.2, max_factor = 1.2)
p.random_distortion(probability = 0.5, grid_width = 3, grid_height = 3, magnitude = 4)
p.sample(1000)