# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:20:12 2016
centroid,area,length of shapes

geometry1.py
@author: skraft
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_segmentation import image_segmentation

img = cv2.imread('segmented.jpg')

(y,x,clrs) = img.shape
img_seg = image_segmentation(img)

color1 =['r','g','b','c','m','y']
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

#fig = plt.figure()
#plt.title('Original Image')
#plt.imshow(img,cmap='Greys_r')

#Find Edges using canny
img_edges = cv2.Canny(img_seg.copy(),30,200)
fig2 = plt.figure()
plt.title('Canny Edge Detection')
plt.imshow(img_edges,cmap='Greys_r')



contours, hierarchy = cv2.findContours(img_seg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

c_x = np.zeros(len(contours))
c_y = np.zeros(len(contours))
c_w = np.zeros(len(contours))
c_h = np.zeros(len(contours))
c_area = np.zeros(len(contours))
c_centroid = np.zeros(len(contours))


fig3 = plt.figure()
plt.imshow(img,cmap='Greys_r')
plt.title('Plot Found Contour Rectangles')
for n,contour in enumerate(contours):
    (c_x[n],c_y[n],c_w[n],c_h[n]) = cv2.boundingRect(contour)
    c_area[n] = cv2.contourArea(contour)
    rect_plt = plt.Rectangle((c_x[n],c_y[n]),c_w[n],c_h[n], color='r',fill=False)
    plt.gca().add_patch(rect_plt)

#img = cv2.drawContours(img, contours, 3, (0,255,0), 3)