import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import math

# Load Data from Directory
img_dir = "directory"
files = glob.glob(img_dir)
data = []
for f in files:
    img = cv2.imread(f)
    data.append(img)


# Augment dataset
#
# Essentially: increase length of dataset by adding altered images 
# Populate tensor with images that are:
#   - Flipped horizontally
#   - Randomly cropped & resized to the original size
#   - Scaling the R, G, and B channels by a value between [0.6 and 1.0]
#   - A combination of the above three
#
aug_data = []
# for img in data:
    

# Convert data to L*a*b* color space (Luminance+Chrominance Space)
dataLAB=[]
for img in aug_data:
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# Batch Normalization
#
# Use SpatialBatchNormalization layer in Torch
# Divide DS into minibatch of 10 images each
#


# *Build* + *Train* a Regressor
#
# Input: The L* channel (grayscale)
# Output: Mean chrominance


# Colorize Image


# Move to CUDA CuNN


# EC: TANH

# EC: CHange no. of feature maps for interior NNs

# EC: Review Paper

