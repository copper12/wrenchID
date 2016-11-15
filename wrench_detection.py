# -*- coding: utf-8 -*-
"""
wrench_detection.py

Conversion of MATLAB code to python 2.7
Stefan Kraft, November 15, 2016
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imtools as imt
from image_segmentation_length import *
from image_segmentation import *
from back_ground_remove import *

im=cv2.imread('2.jpg');
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

(y,x,clrs) = im.shape
img_hou = im[0:y/2, 0:x] # Cropping Image (matlab line 16)

plt.imshow(img_hou)

# Alter contrast, brightness

img2_hou = imt.imadjust(img_hou)
img2 = imt.imadjust(im)

fig = plt.figure()
plt.imshow(img2_hou)


# Remove Background

img_remove_hou = back_ground_remove(img2_hou)
img_remove = back_ground_remove(img2)

img_seg_hou = image_segmentation(img_remove_hou)
img_seg = image_segmentation_length(img_remove_hou)

fig = plt.figure()
plt.imshow(img_seg_hou)

color1 =['r','g','b','c','m','y']
radius =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

fig2 = plt.figure()
fig2.imshow(im)