import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import torch as t
import torch.nn as nn
import torch.nn.functional as Fn
import math
import random as rd

# TODO: HOW MAKE TENSOR
# TODO: HOW MAKE CNN
# TODO: HOW FILL TENSOR
# Load Data from Directory
img_dir = "face_images/*.jpg"
files = glob.glob(img_dir)
data = []
for f in files:
    img = cv2.imread(f)
    data.append(img)


# data = tf.convert_to_tensor(data)


# Augment dataset
#
# Essentially: increase length of dataset by adding altered images 
# Populate tensor with images that are:
#   - Flipped horizontally
#   - Randomly cropped & resized to the original size
#   - Scaling the R, G, and B channels by a value between [0.6 and 1.0]
#   - A combination of the above three
#

# Scale Image BGR by between 0.6 and 1.0
def aug_scale_rgb(image):
    random_scale = rd.uniform(0.6, 1.0)
    for p_i in range(len(image[0])):
        for p_j in range(len(image)):
            for c in range(3):
                # Scale the BGR by a random amount
                image[p_i][p_j][c] = int(image[p_i][p_j][c] * random_scale)
    return image


# Crop Image and resize
def aug_crop(image):
    x_init = rd.randint(0, 64)
    y_init = rd.randint(0, 64)
    random_val = rd.randint(64, 128 - x_init)
    w = random_val
    h = random_val
    crop = image[y_init:y_init + h, x_init:x_init + w]
    resize_w = int(len(crop[0]) * (128 / w))
    resize_h = int(len(crop) * (128 / h))
    resize_scale = (resize_w, resize_h)
    return cv2.resize(crop, resize_scale, interpolation=cv2.INTER_AREA)


# Flip image horizontally
def aug_flip(image):
    return cv2.flip(image, 1)


scale = 10
# aug_data = tf.fill([len(data) * scale, 128, 128, 3], 0)
curr = 0
# For each image in data
for img in data:
    # Do this n times for each image
    cv2.imshow("original", img)
    cv2.waitKey(0)
    resized = aug_crop(img)
    flipped = aug_flip(img)
    bgr = aug_scale_rgb(img)
    # for i in range(1, scale):
    # curr += 1

# Convert data to L*a*b* color space (Luminance+Chrominance Space)
dataLAB = []
i = 0
# TODO: change "data" to "aug_data
for img in data:
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    dataLAB.append(imgLAB)

print(dataLAB[0])

# Batch Normalization
#
# Use SpatialBatchNormalization layer in Torch
# Divide DS into minibatch of 10 images each
#


# *Build* + *Train* a Regressor
#
# Input: The L* channel (grayscale)
# Output: Mean chrominance


# SET THESE
batch_size = 1
epoch = 1


# TODO: WTH IS THIS.?
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1)
        self.conv5 = nn.Conv2d(16, 8, 3, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1)
        self.conv7 = nn.Conv2d(4, 2, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = Fn.relu(x)
        x = self.conv2(x)
        x = Fn.relu(x)
        x = self.conv3(x)
        x = Fn.relu(x)
        x = self.conv4(x)
        x = Fn.relu(x)
        x = self.conv5(x)
        x = Fn.relu(x)
        x = self.conv6(x)
        x = Fn.relu(x)
        x = self.conv7(x)
        x = Fn.relu(x)

# Forward
# Compute Loss
# Backward
# Update Weights
# Testing


# Colorize Image


# Move to CUDA CuNN


# EC: TANH

# EC: Change no. of feature maps for interior NNs

# EC: Review Paper

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
