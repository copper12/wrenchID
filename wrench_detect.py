#!/usr/bin/python

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

from imadjust import imadjust
from stretchlim import stretchlim
from back_ground_remove import back_ground_remove
from image_segmentation_length import image_segmentation_length


#Read image
im = cv2.imread('pipewrench.jpeg',1)

lims = stretchlim(im)
img2 = imadjust(im,lims)
img_remove = back_ground_remove(img2)

cv2.imshow('imadjust output',img2)
cv2.imshow('background removal output',img_remove)

img_seg=image_segmentation_length(img_remove);
cv2.imshow('Image Segmentation Output',img_seg)
cv2.waitKey(0)
