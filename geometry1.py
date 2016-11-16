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

#Find Edges using canny, not necessary
img_edges = cv2.Canny(img_seg.copy(),30,200)
fig2 = plt.figure()
plt.title('Canny Edge Detection')
plt.imshow(img_edges,cmap='Greys_r')


# Find Contours
contours, hierarchy = cv2.findContours(img_seg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Initialize contour region variables
c_x = np.zeros(len(contours))
c_y = np.zeros(len(contours))
c_w = np.zeros(len(contours))
c_h = np.zeros(len(contours))
c_area = np.zeros(len(contours))
c_length = np.zeros(len(contours))
c_centroidx = np.zeros(len(contours))
c_centroidy = np.zeros(len(contours))

moments = []

fig3 = plt.figure()
plt.imshow(img,cmap='Greys_r')
plt.title('Plot Found Contour Rectangles')
for n,contour in enumerate(contours):
    (c_x[n],c_y[n],c_w[n],c_h[n]) = cv2.boundingRect(contour)
    c_area[n] = cv2.contourArea(contour)
        
    minarearect = cv2.minAreaRect(contour)
    (wdth,lgth) = minarearect[1]
    c_length[n] = np.max([wdth,lgth])
    moments = cv2.moments(contour)
    
    if(int(moments['m00']) is not 0):
        c_centroidx[n] = int(moments['m10']/moments['m00'])
        c_centroidy[n] = int(moments['m01']/moments['m00'])


id_len = np.argsort(c_length)

if(len(id_len)>6):
    #Found more than 6 regions, n_wrench = 6
    n_wr = 6
else:
    n_wr = len(id_len)

# Take the last 6 (or num wrenches) items in the arrays, as 
# sorting is from smallest to largest
c_length2 = c_length[id_len[-n_wr::]]
c_area2 = c_area[id_len[-n_wr::]]
c_centroidx2 = c_centroidx[id_len[-n_wr::]]
c_centroidy2 = c_centroidy[id_len[-n_wr::]]

for n in id_len[-6::]:
    rect_plt = plt.Rectangle((c_x[n],c_y[n]),c_w[n],c_h[n], color='r',fill=False)
    plt.gca().add_patch(rect_plt)
    
    centroid_plt = plt.Circle((c_centroidx[n],c_centroidy[n]), 
                              2, color=color1[2])
    plt.gca().add_patch(centroid_plt)