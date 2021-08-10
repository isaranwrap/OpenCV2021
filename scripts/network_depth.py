from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import skimage.measure
import os
 
lrcheck = False
extended = False
subpixel = False
 
 
parent = Path(__file__).parent.absolute()
imagePath = str((Path(parent).parent.absolute() / Path('data/set2')).resolve().absolute())
nnPath = str((Path(parent).parent.absolute() / Path('models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute())
 
#instantiate pipeline
pipeline = dai.Pipeline()
 
#define sources and outputs
cam_rgb = pipeline.createColorCamera()
cam_left = pipeline.createMonoCamera()
cam_right = pipeline.createMonoCamera()
cam_stereo = pipeline.createStereoDepth()
nn = pipeline.createMobileNetDetectionNetwork()
manip = pipeline.createImageManip()
 
xout_rgb = pipeline.createXLinkOut()
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
xout_disp = pipeline.createXLinkOut()
xout_depth = pipeline.createXLinkOut()
xout_manip = pipeline.createXLinkOut()
xout_nn = pipeline.createXLinkOut()
 
xout_rgb.setStreamName("rgb")
xout_left.setStreamName("left")
xout_right.setStreamName("right")
xout_disp.setStreamName("disparity")
xout_depth.setStreamName("depth")
xout_manip.setStreamName("manip")
xout_nn.setStreamName("nn")
 
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
 
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
 
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
 
cam_stereo.setInputResolution(640, 400)
cam_stereo.initialConfig.setConfidenceThreshold(240)
cam_stereo.setOutputRectified(True)
cam_stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
if lrcheck or extended or subpixel:
   cam_stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
else:
   cam_stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
cam_stereo.setLeftRightCheck(lrcheck)
cam_stereo.setExtendedDisparity(extended)
cam_stereo.setSubpixel(subpixel)
 
nn.setConfidenceThreshold(.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
 
#ImageManip takes frame and preps it for NN
manip.initialConfig.setResize(300,300)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
 
cam_stereo.rectifiedRight.link(manip.inputImage)
 
manip.out.link(nn.input)
cam_rgb.preview.link(xout_rgb.input)
cam_left.out.link(cam_stereo.left)
cam_right.out.link(cam_stereo.right)
cam_stereo.disparity.link(xout_disp.input)
cam_stereo.depth.link(xout_depth.input)
cam_right.out.link(xout_right.input)
cam_left.out.link(xout_left.input)
manip.out.link(xout_manip.input)
nn.out.link(xout_nn.input)
 
#Mobilenet SSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus","car", "cat", "chair", "cow","diningtable", "dog", "horse", "motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"]
 
with dai.Device(pipeline) as device:
   device.startPipeline()
   qSize = 4
 
   rgb_q = device.getOutputQueue(name="rgb", maxSize=qSize, blocking=False)
   left_q = device.getOutputQueue(name="left", maxSize=qSize, blocking=False)
   right_q = device.getOutputQueue(name="right", maxSize=qSize, blocking=False)
   disparity_q = device.getOutputQueue(name="disparity", maxSize=qSize, blocking=False)
   depth_q = device.getOutputQueue(name="depth", maxSize=qSize, blocking=False)
   manip_q = device.getOutputQueue(name="manip", maxSize=qSize, blocking=False)
   nn_q = device.getOutputQueue(name="nn", maxSize=qSize, blocking=False)
 
   dispImage = None
   detections = []
   offsetX = (cam_right.getResolutionWidth() - cam_right.getResolutionHeight())
   croppedFrame = np.zeros((cam_right.getResolutionHeight(), cam_right.getResolutionHeight()))
 
   def frameNorm(frame, bbox):
       normVals = np.full(len(bbox), frame.shape[0])
       normVals[::2] = frame.shape[1]
       return (np.clip(np.array(bbox),0,1)*normVals).astype(int)
 
   disparityMultiplier = 255/cam_stereo.getMaxDisparity() #used to normalize image
 
   shot = 0
   food = 'tomato'
   while True:
  
       rgb = rgb_q.tryGet()
       if rgb is not None:
           rgb = rgb_q.get()
           cv2.imshow("rgb", rgb.getCvFrame())
 
       left = left_q.tryGet()
       if left is not None:
           left = left_q.get()
           cv2.imshow("left", left.getFrame())
 
       right = right_q.tryGet()
       if right is not None:
           right = right_q.get()
           cv2.imshow("right", right.getFrame())
 
       manips = manip_q.tryGet()
       if manips is not None:
           manips = manip_q.get()
           cv2.imshow("manip", manips.getCvFrame())
 
       disparity = disparity_q.tryGet()
       if disparity is not None:
           disparity = disparity_q.get()
           dispImage = (disparity.getCvFrame()*disparityMultiplier).astype(np.uint8) #multiply disparity out to full range
           cv2.imshow("disparity", dispImage)
           for detection in detections:
               bbox = frameNorm(croppedFrame,(detection.xmin, detection.ymin, detection.xmax, detection.ymax))
               bbox[::2] += offsetX
               cv2.rectangle(dispColorImage, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),30)
               cv2.putText(dispColorImage, labelMap[detection.label], (bbox[0]+10, bbox[1]+20), cv2.FONT_HERSHEY_TRIPLEX, .5, 255)
               cv2.putText(dispColorImage, f"/int(detection.confidence*100)/%", (bbox[0]+10, bbox[1]+40), cv2.FONT_HERSHEY_TRIPLEX, .5, 255)
           #dispImage = skimage.measure.block_reduce(dispImage, (2,2), np.min)
           dispColorImage = cv2.applyColorMap(dispImage, cv2.COLORMAP_HOT)
           cv2.imshow("disparity_color", dispColorImage)
      
       depth = depth_q.tryGet()
       if depth is not None:
           depth = depth.getFrame().astype(np.uint8)
           np.where(depth<20,0,depth)
           np.where(depth>5000,0,depth)
           depthImage = skimage.measure.block_reduce(depth, (2,2), np.max)
           cv2.imshow("depth", depthImage)
           #depth = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
           #depth = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)
           #cv2.imshow("depth", depth)
 
       det = nn_q.tryGet()
       if det is not None:
           detections = det.detections
 
       if cv2.waitKey(1) == ord('q'):
           cv2.destroyAllWindows()
           break 
      
       if cv2.waitKey(1) == ord('n'):
           shot += 1
           print('Shot incremented to {}'.format(shot))
 
       if cv2.waitKey(1) == ord('p'):
           print('Saved picture of {}, shot: {}'.format(food, shot))
           cv2.imwrite(os.path.join(imagePath, 'rgb/{}{}.png'.format(food, shot)),rgb.getCvFrame())
           cv2.imwrite(os.path.join(imagePath, 'left/{}{}.png'.format(food, shot)),left.getFrame())
           cv2.imwrite(os.path.join(imagePath, 'right/{}{}.png').format(food, shot),right.getFrame())
           cv2.imwrite(os.path.join(imagePath, 'disparity/{}{}.png'.format(food, shot)), dispColorImage)
           cv2.imwrite(os.path.join(imagePath, 'depth/{}{}.png'.format(food, shot)),depth)