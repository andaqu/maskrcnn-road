import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import tensorflow as tf
sys.path.insert(1, r'C:\Users\andrew\Desktop\home\uni\vision2\ARI3129-Advanced-Vision\maskrcnn-road\samples\custom')
import custom

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# CHANGE THIS PATH
custom_WEIGHTS_PATH = r'C:\Users\andrew\Desktop\home\uni\vision2\ARI3129-Advanced-Vision\maskrcnn-road\mask_rcnn_object_0080.h5'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
  
# Load test dataset
dataset = custom.CustomDataset()
dataset.load_custom(custom_DIR, "test")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset.image_ids, len(dataset.image_ids))

# Instanciate arrays to create our metrics
APs = []
precisions_arr = []
recalls_arr = []
overlaps_arr = []
class_ids_arr = []
scores_arr = []

for id in dataset.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_ids[id], use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'], 1)
    # Append AP to AP array
    APs.append(AP)
    
    # Append precisions
    for precision in precisions:
        precisions_arr.append(precision)
    
    # Append recalls
    for recall in recalls:
        recalls_arr.append(recall)
    
    # Append overlaps
    for overlap in overlaps:
        overlaps_arr.append(overlap)
    
    # Append class_ids
    for class_id in r["class_ids"]:
        class_ids_arr.append(class_id)
    
    # Append scores 
    for score in r["scores"]:
        scores_arr.append(score)
    
print("mAP: ", np.mean(APs))
# car: 0.6228839671702139