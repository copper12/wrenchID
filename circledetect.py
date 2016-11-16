# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:54:04 2016

circledetect.py

@author: skraft
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('segmented.jpg')

(y,x,clrs) = im.shape
img_seg_hou = np.copy(im[0:y/2, 0:x])
img_seg_hou = cv2.cvtColor(img_seg_hou, cv2.COLOR_BGR2GRAY)

color1 =['r','g','b','c','m','y']
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

circles = cv2.HoughCircles(img_seg_hou, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

fig = plt.figure()
plt.imshow(img_seg_hou,cmap='Greys_r')
for n,circle in enumerate(circles[0,:,:]):
    circ_plt = plt.Circle((circle[0],circle[1]), 
                         circle[2], color=color1[1], fill=False)
    cntr_plt = plt.Circle((circle[0],circle[1]), 
                         1, color=color1[2])
    plt.gca().add_patch(circ_plt)
    plt.gca().add_patch(cntr_plt)