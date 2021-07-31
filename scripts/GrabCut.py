import numpy as np
import argparse
import time
import cv2
import os

#construct arg parser, parse args for cmd prompt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	default=os.path.sep.join(["images", "cheeseburger-3.jpg"]), #cheeseburger.jpg
	help="path to input image that we'll apply GrabCut to")
ap.add_argument("-c", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())

#load image, allocate memory for output mask
image = cv2.imread(args["image"])
mask = np.zeros(image.shape[:2], dtype="uint8")

#Does worse with larger images
rect = (13,20, 234, 218) #box around foreground, CNN handles this

#allocate memory for foreground,background matrix
fgModel = np.zeros((1,65), dtype="float")
bgModel = np.zeros((1,65), dtype="float")

#apply GrabCut
start = time.time()
mask, bgModel, fgModel = cv2.grabCut(image, mask, rect, bgModel, fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)
end = time.time()
print("APPLYING GRABCUT TOOK {:.2f} SECONDS".format(end-start))

#Create set for grabcut foreground/background distinctions
values = {
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
}

for (name, value) in values:
    print("MASK FOR {}".format(name))
    valMask = (mask==value).astype("uint8")*255
    cv2.imshow(name, valMask)
    cv2.waitKey(0)

#Background pixel=0, Foreground pixel=1
outputMask = np.where((mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD),0,1)
print(outputMask)
leftmost = (outputMask.shape[1],0)
rightmost = (0,0)

for idx,elem in np.ndenumerate(outputMask):
    if elem == 1:
        if idx[0] < leftmost[0]:
            leftmost = idx
        elif idx[0] > rightmost[0]:
            rightmost = idx

outputMask = (outputMask*255).astype("uint8")

area = cv2.countNonZero(outputMask)
percent = area/(image.shape[0]*image.shape[1])*100
print("AREA OF FOOD IN {} is {} pixels and {} percent of screen".format(args["image"],area,percent))

#overlap between mask and original for final output
output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.circle(output, leftmost, 8, (0, 100, 0),10)
cv2.circle(output, rightmost, 8, (0, 100, 0),10)

cv2.imshow("INPUT",image)
cv2.imshow("GRABCUT MASK",outputMask)
cv2.imshow("GRABCUT OUTPUT",output)
cv2.waitKey()