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

from imadjust import imadjust
from stretchlim import stretchlim
from back_ground_remove import back_ground_remove
from image_segmentation import image_segmentation
from image_segmentation_length import image_segmentation_length

im=cv2.imread('2.jpg',1);
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

(y,x,clrs) = im.shape
img_hou = np.copy(im[0:y/2, 0:x]) # Cropping Image (matlab line 16)

fig1 = plt.figure()
plt.title('Initial Image')
plt.imshow(im)

# Alter contrast, brightness

lims = stretchlim(im)
img2 = np.copy(imadjust(im,lims))
lims_hou = stretchlim(img_hou)
img2_hou = np.copy(imadjust(img2,lims_hou))

fig2 = plt.figure()
plt.title('imadjust')
plt.imshow(img2)

# Remove Background

img_remove_hou = np.copy(back_ground_remove(img2_hou))
img_remove = np.copy(back_ground_remove(img2))

fig3 = plt.figure()
plt.title('Remove Background')
plt.imshow(img_remove_hou)

img_seg_hou = image_segmentation(img_remove_hou)
img_seg = image_segmentation_length(img_remove_hou)

fig4 = plt.figure()
plt.title('Image Segmentation')
plt.imshow(img_seg_hou)

# img_edge = edge(img_seg_hou,'canny');
# Nowhere else in the matlab code was this used, commented out

color1 =['r','g','b','c','m','y']
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

fig5 = plt.figure()
sbplt = fig5.add_subplot(111)
sbplt.imshow(im)

# Doesn't get any circles, due to poor image processing
for radius in radius_array:
     cicles = cv2.HoughCircles(img_seg_hou, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
     
#    Need to plot centers as in matlab code used previously
#      plot(centers(:,1),centers(:,2),'+','LineWidth',2,'Color',color1(i));
#  for j=1:size(centers,1)
#    theta = 0 : 0.01 : 2*pi;
#    x = radius(i) * cos(theta) + centers(j,1);
#    y = radius(i) * sin(theta) + centers(j,2);
#    plot(x, y,color1(i), 'LineWidth', 2);
#  end     
     
#     print(x)
     
color1_l = np.fliplr([color1]);


#s=regionprops(img_seg,{'Centroid','BoundingBox','Area'});
# Implement Detect Length and Area:

(contours, hierarchy) = cv2.findContours(img_seg,1,2)

numobjs = len(contours)
x = np.zeros(numobjs)
y = np.zeros(numobjs)
w = np.zeros(numobjs)
h = np.zeros(numobjs)
areas = np.zeros(numobjs)
lengths = np.zeros(numobjs)

for n,contour in enumerate(contours):
    (x[n],y[n],w[n],h[n]) = cv2.boundingRect(contour)
    
    areas[n] = w[n]*h[n]
    b_w = 1 # Bounding Box; needs implementation
    lengths[n] = np.sqrt(w[n]**2 + h[n]**2)
    # Need to implement lines 67 to 130, find area, length, centriod
    # of rectangles, plot on original graph
    
if(len(lengths[lengths>0])<6):
    mm= len(lengths[lengths>0])
else:
    mm = 6 # Not sure why this is the variable name, a lot of code
            # goes into determining some of the parameters in this section


x_length = np.zeros(mm)
x_area = np.zeros(mm)
y_area = np.zeros(mm)

for n in range(mm):
    cen = [x[n],y[n]]  # Need actual centroid location
    bb = [x[n],y[n],x[n]+w[n],y[n]+h[n]] # Need actual bounding box
    x_area[n] = x[n]*w[n] #Incorrect, placeholder
    y_area[n] = y[n]*h[n] # Incorrect, placeholder
    
positions = np.zeros((3,mm))
positions[0,:] = np.arange(mm)
positions[1,:] = x_area[0:mm]
positions[2,:] = x_length[0:mm:]

e=40 # Pixels, not sure what this does.

#Voting goes here, not sure how it works or why.
    
 


