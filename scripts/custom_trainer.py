import os
<<<<<<< HEAD
import random
import sys
import json
import datetime
from cv2 import data
=======
import sys
import json
import datetime
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
import numpy as np
import skimage.draw
import cv2
from pathlib import Path
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

#https://github.com/matterport/Mask_RCNN/issues/2637
#https://github.com/leekunhee/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb

<<<<<<< HEAD
__file__ = '/Users/Praveens/Desktop/ishan/OpenCV2021/scripts/custom_trainer.py'
parent = Path(__file__).parent.absolute()
sys.path.append(parent)
#sys.path.append('/Users/Praveens/Desktop/ishan/Mask_RCNN')

=======
parent = Path(__file__).parent.absolute()

sys.path.append(parent)
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

<<<<<<< HEAD
datasetTrain = str((Path(parent).parent.absolute() / Path('data/data/rgb-cam1')).resolve().absolute())#FIX
datasetVal = str((Path(parent).parent.absolute() / Path('data/data/rgb-cam2')).resolve().absolute())#FIX
weightsPath = str((Path(parent).parent.absolute() / Path('models/mask_rcnn_coco.h5')).resolve().absolute())
jsonPath = str((Path(parent).parent.absolute() / Path('data/data/Mask-RCNN_json.json')).resolve().absolute())
=======
datasetTrain = str((Path(parent).parent.absolute() / Path('data/rgb-cam1')).resolve().absolute())
datasetVal = str((Path(parent).parent.absolute() / Path('data/rgb-cam2')).resolve().absolute())
weightsPath = str((Path(parent).parent.absolute() / Path('models/mask_rcnn_coco.h5')).resolve().absolute())
jsonPath = str((Path(parent).parent.absolute() / Path('data/rgb-cam1/Mask-RCNN_json.json')).resolve().absolute())
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
logDir = str((Path(parent).parent.absolute() / Path('logs')).resolve().absolute())

'''
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
'''

<<<<<<< HEAD
foods = ['CinammonRaisinBagel','OrangeJuice','Steak','Kiwi','Cookie']
=======
foods = ['banana', 'egg','kiwi', 'potato']
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
num_foods = len(foods)

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "foods"

<<<<<<< HEAD
    # Adjust down if you use a smaller GPU.
    # can increase img/gpu with smaller pics
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
=======
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # can increase img/gpu with smaller pics
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350

    # Number of classes (including background)
    NUM_CLASSES = 1 + num_foods  # Background + classes

<<<<<<< HEAD
    #IMAGE_MIN_DIM = 128
    #IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small-->might have to change for food!
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    #TRAIN_ROIS_PER_IMAGE

    #VALIDATION_STEPS = 5

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = .8
=======
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350

config = CustomConfig()
config.display()

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

<<<<<<< HEAD
    def load_custom(self, dataset_dir, subset):

        # Add classes. We have only one class to add.
        for idx, food in enumerate(foods):
          print(idx)
          self.add_class("object", idx+1, foods[idx])

        assert subset in ["train", "val"]
=======
    def load_custom(self, dataset_dir):

        # Add classes. We have only one class to add.
        self.add_class("object", 1, foods[0])
        self.add_class("object", 2, foods[1])
        self.add_class("object", 3, foods[2])
        self.add_class("object", 4, foods[3])
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(jsonPath))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
<<<<<<< HEAD
            
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]  
=======
            polygons = [r['shape_attributes'] for r in a['regions']] 
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {foods[0]: 1,foods[1]: 2, foods[2]: 3, foods[3]: 4}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
<<<<<<< HEAD
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(datasetTrain, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(datasetVal, "val")
    dataset_val.prepare()

    print("Training network heads!")
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')

    return dataset_train, dataset_val

'''
image_ids = np.random.choice(dataset_train.image_ids,4)
for image_id in image_ids:
    image = dataset_train.image_reference(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
'''

=======

"""Train the model."""
# Training dataset.
dataset_train = CustomDataset()
dataset_train.load_custom(datasetTrain)
dataset_train.prepare()

# Validation dataset
dataset_val = CustomDataset()
dataset_val.load_custom(datasetVal)
dataset_val.prepare()


config = CustomConfig()
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logDir)

weights_path = weightsPath
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

<<<<<<< HEAD
dataset_train, dataset_val = train(model)
=======
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=1,
            layers='heads')
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()

<<<<<<< HEAD
model2 = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=logDir)
model2.load_weights(weights_path, by_name=True)
=======
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=logDir)

>>>>>>> dffe70f9199a4531562558483b81083aa8f92350

image_id = random.choice(dataset_val.image_ids)
og_image, img_mta, class_id, bbox, mask = modellib.load_image_gt(dataset_val, inference_config, image_id)

log("original_image", og_image)
log("image_meta", img_mta)
log("gt_class_id", class_id)
log("gt_bbox", bbox)
log("gt_mask", mask)

<<<<<<< HEAD
visualize.display_instances(og_image, bbox, mask, class_id, 
=======
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
>>>>>>> dffe70f9199a4531562558483b81083aa8f92350
                            dataset_train.class_names, figsize=(8, 8))

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))