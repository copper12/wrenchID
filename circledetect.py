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

(y,x,clrs) = img.shape
img_seg_hou = np.copy(img[0:y/2, 0:x])
img_seg_hou = cv2.cvtColor(img_seg_hou, cv2.COLOR_BGR2GRAY)

color1 =['r','g','b','c','m','y']
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

circles = cv2.HoughCircles(img_seg_hou, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

center_x = np.zeros(len(circles[0]))
center_y = np.zeros(len(circles[0]))
radius = np.zeros(len(circles[0]))

fig = plt.figure()
plt.imshow(img_seg_hou,cmap='Greys_r')
#for n,circle in enumerate(circles[0,:,:]):
center_x = circles[0,:,0] 
center_y = circles[0,:,1]
radius = circles[0,:,2]

id_r = np.argsort(radius)

if(len(id_r)>6):
    #Found more than 6 circles, n_wrench = 6
    n_wr = 6
else:
    n_wr = len(id_r)
    
center_x2 = center_x[id_r[0:n_wr]]
center_y2 = center_y[id_r[0:n_wr]]
radius_2 = radius[id_r[0:n_wr]]

for n in range(n_wr):
    circ_plt = plt.Circle((center_x2[n],center_y2[n]), 
                          radius[n], color=color1[1], fill=False)
    cntr_plt = plt.Circle((center_x2[n],center_y2[n]), 
                         1, color=color1[2])
    plt.gca().add_patch(circ_plt)
    plt.gca().add_patch(cntr_plt)