import os
import numpy as np
import mrcnn.config
import mrcnn.utils
import skimage.io
import cv2
import random
import csv
from mrcnn.model import MaskRCNN
from pathlib import Path

# TensorFlow - Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

# Filter a list of Mask R-CNN detection results to get only the detected persons
def get_person_boxes(boxes, class_ids):
    person_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a person, skip it
        if class_ids[i] in [1]:
            person_boxes.append(box)

    return np.array(person_boxes)

# Logs the bounding boxes in a .csv file
def logger(filepath,results):
    indexes = [pos for pos, char in enumerate(filepath) if char == '/']
    filenameCSV = filepath[indexes[-1] + 1:(-5)] + '.csv'
    filepathCSV = os.path.join(filepath[0:indexes[-2]], "logs", filenameCSV)

    print(filepathCSV)

    with open(filepathCSV, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["y1","x1","y2","x2"])
        for boxes in results:
            writer.writerow(boxes)

# Root directory of the project
#ROOT_DIR = Path(".")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save logs
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
# parked_car_boxes = None

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
imagepath = os.path.join(IMAGE_DIR, random.choice(file_names))
image = skimage.io.imread(imagepath)

# Run the image through the Mask R-CNN model to get results.
# results is a python dictionary
results = model.detect([image], verbose=0)

# Mask R-CNN assumes we are running detection on multiple images.
# We only passed in one image to detect, so only grab the first result.
r = results[0]
print("Number of objects detected:" + str(len(r['rois'])) )

# The r variable will now have the results of detection:
# - r['rois'] are the bounding box of each detected object
# - r['class_ids'] are the class id (type) of each detected object
# - r['scores'] are the confidence scores for each detection
# - r['masks'] are the object masks for each detected object (which gives you the object outline)

# Filter the results to only grab the persons bounding boxes
person_boxes = get_person_boxes(r['rois'], r['class_ids'])
print("Persons found in image:" + str(len(person_boxes)))

logger(imagepath,person_boxes)

'''
# Draw each box on the frame
for box in person_boxes:
    print("Person: ", box)

    y1, x1, y2, x2 = box

    # Draw the box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('Imagem', image)
cv2.waitKey()
'''


    

