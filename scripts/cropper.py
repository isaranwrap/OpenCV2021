import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

imageIndex = 0
save = False
root = 'img'
imageFolder = '../data/data/rgb-cam2'
outFolder = '../data/data/clean'
imageName = os.listdir(imageFolder)[imageIndex]
img = cv2.imread(os.path.join(imageFolder, imageName))
rect_color = (255, 0, 0) # blue
rect_linewidth = 1
cropSize = 80
crop_x1, crop_x2 = None, None
crop_y1, crop_y2 = None, None

def on_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        global crop_x1, crop_x2, crop_y1, crop_y2
        crop_x1, crop_x2 = x - cropSize, x + cropSize
        crop_y1, crop_y2 = y - cropSize, y + cropSize
        cv2.rectangle(img, (crop_x1-1, crop_y1-1), (crop_x2+1,crop_y2+1), rect_color, rect_linewidth)
        print(crop_x1, crop_x2, crop_y1, crop_y2)



cv2.namedWindow(root)
cv2.setMouseCallback(root, on_mouse_event)

while True:
    code = cv2.waitKey(1)
    
    cv2.imshow(root, img)
    
    #if crop_x1 is not None:
    #    print(crop_x1, crop_x2, crop_y1, crop_y2)
        #cv2.imshow("root", img[crop_x1:crop_x2, crop_y1:crop_y2])
    
    if code == ord('n'):
        imageIndex += 1 
        imageName = os.listdir(imageFolder)[imageIndex]
        img = cv2.imread(os.path.join(imageFolder, imageName))
    
    if code == ord('p'):
        imageIndex -= 1
        imageName = os.listdir(imageFolder)[imageIndex]
        img = cv2.imread(os.path.join(imageFolder, imageName))

    if code == ord('s'):
        save = not save
        print('Save switched to {}'.format(save))
    if save:
        cv2.imwrite(os.path.join(outFolder, imageName), img[crop_y1:crop_y2, crop_x1:crop_x2])


cv2.destroyAllWindows()