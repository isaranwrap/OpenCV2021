import cv2
import numpy as np
import depthai as dai
import time

lrcheck = False
extended = False
subpixel = False

#nnPath = str((Path(__file__).parent / Path('models/...')).resolve().absolute())

#instantiate pipeline
pipeline = dai.Pipeline()

#define sources and outputs
cam_rgb = pipeline.createColorCamera()
cam_left = pipeline.createMonoCamera()
cam_right = pipeline.createMonoCamera()
cam_stereo = pipeline.createStereoDepth()

xout_rgb = pipeline.createXLinkOut()
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
xout_disp = pipeline.createXLinkOut()
xout_depth = pipeline.createXLinkOut()
#xout_rectif_left = pipeline.createXLinkOut()
#xout_rectif_right = pipeline.createXLinkOut()

xout_rgb.setStreamName("rgb")
xout_left.setStreamName("left")
xout_right.setStreamName("right")
xout_disp.setStreamName("disparity")
xout_depth.setStreamName("depth")

#xout_rectif_left.setStreamName("rectified_left")
#xout_rectif_right.setStreamName("rectified_right")

#set properties
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

cam_stereo.setInputResolution(640, 400)
cam_stereo.initialConfig.setConfidenceThreshold(200)
cam_stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
if lrcheck or extended or subpixel:
    cam_stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
else:
    cam_stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
#cam_stereo.setOutputRectified(True)
cam_stereo.setLeftRightCheck(lrcheck)
cam_stereo.setExtendedDisparity(extended)
cam_stereo.setSubpixel(subpixel)

#link inputs to outputs
cam_rgb.preview.link(xout_rgb.input)

cam_left.out.link(cam_stereo.left)

cam_right.out.link(cam_stereo.right)

cam_stereo.syncedLeft.link(xout_left.input)
cam_stereo.syncedRight.link(xout_right.input)
cam_stereo.disparity.link(xout_disp.input)
cam_stereo.depth.link(xout_depth.input)

#cam_stereo.rectifiedLeft.link(xout_rectif_left.input)
#cam_stereo.rectifiedRight.link(xout_rectif_right.input)
'''
def change_ConfidenceThreshold(x):
    cam_stereo.change_ConfidenceThreshold(cv2.getTrackbarPos('ConfidenceThreshold', 'controls'))

def change_MedianFilter(x):
    if cv2.getTrackbarPos('MedianFilter', 'controls') == 0:
        self
        cam_stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    elif cv2.getTrackbarPos('MedianFilter', 'controls') == 1:
        cam_stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
    elif cv2.getTrackbarPos('MedianFilter', 'controls') == 2:
        cam_stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    else:
        cam_stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

def change_LRCheck(x):
    if cv2.getTrackbarPos('LRCheck', 'controls') == 0:
        cam_stereo.setLeftRightCheck(False)
    else:
        cam_stereo.setLeftRightCheck(True)

def change_ExtendedDisparity(x):
    if cv2.getTrackbarPos('ExtendedDisparity', 'controls') == 0:
        cam_stereo.setExtendedDisparity(False)
    else:
        cam_stereo.setExtendedDisparity(True)

def change_Subpixel(x):
    if cv2.getTrackbarPos('Subpixel', 'controls') == 0:
        cam_stereo.setSubpixel(False)
    else:
        cam_stereo.setSubpixel(True)

cv2.namedWindow('controls')
cv2.createTrackbar('ConfidenceThreshold', 'controls', 0, 255, change_ConfidenceThreshold)
cv2.createTrackbar('MedianFilter', 'controls', 0, 3, change_MedianFilter)
cv2.createTrackbar('LRCheck', 'controls', 0, 1, change_LRCheck)
cv2.createTrackbar('ExtendedDisparity', 'controls', 0, 1, change_ExtendedDisparity)
cv2.createTrackbar('Subpixel', 'controls', 0, 1, change_Subpixel)
'''


with dai.Device(pipeline) as device:
    device.startPipeline()

    #collect information from input channels
    rgb_q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    left_q = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_q = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    disparity_q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    depth_q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    #nn_q = device.getOutputQueue(name="nn", maxSize=8, blocking=False)
    #rectRight_q = device.getOutputQueue(name="rectified_right", maxSize=4, blocking=False)
    #rectLeft_q = device.getOutputQueue(name="rectified_left", maxSize=4, blocking=False)

    disparityMultiplier = 255/cam_stereo.getMaxDisparity() #used to normalize image

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

        disparity = disparity_q.tryGet()
        if disparity is not None:
            disparity = disparity_q.get()
            dispImage = (disparity.getCvFrame()*disparityMultiplier).astype(np.uint8) #multiply disparity out to full range
            cv2.imshow("disparity", dispImage)
            dispColorImage = cv2.applyColorMap(dispImage, cv2.COLORMAP_JET)
            cv2.imshow("disparity_color", dispColorImage)

        '''
        depth = depth_q.tryGet()
        if depth is not None:
            depth = depth_q.get()
            depthImage = (depth.getCvFrame().astype(np.uint16))
            depthImage = (depthImage/256).astype(np.uint8)
            cv2.imshow("depth", depthImage)#wonder if astype(np.uint16) means that we can get even more depth "resolution"
            depthColorImage = cv2.applyColorMap(depthImage, cv2.COLORMAP_JET)
            cv2.imshow("depth_color", depthColorImage)
        '''

        '''
        rectLeft = rectLeft_q.tryGet()
        if rectLeft is not None:
            rectLeft = rectLeft_q.get()
            rectLeftImage = cv2.flip((rectLeft.getCvFrame()).astype(np.uint8),1)
            cv2.imshow("rectLeft", rectLeftImage)
        '''

        if cv2.waitKey(1) == ord('p'):
            cv2.imwrite('rgb.jpg',rgb.getCvFrame())
            cv2.imwrite('left.jpg',left.getFrame())
            cv2.imwrite('right.jpg',right.getFrame())
            cv2.imwrite('disparity.jpg',dispImage)
            cv2.imwrite('disparity_color.jpg',dispColorImage)
            #cv2.imwrite('depth.jpg',depth.getCvFrame().astype(np.uint16))
            cv2.destroyAllWindows()
            break
            
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        time.sleep(0.05)