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
from stretchlim import stretchlim, stretchlim2
from back_ground_remove import back_ground_remove
from image_segmentation import image_segmentation
from image_segmentation_length import image_segmentation_length

plt_flag = 1 # 0: Don't plot
             # 1: Plot using plt
             # 2: Plot using cv2.imshow
lim_type = 1 # 1: Same as Matlab (I think)
             # 2: Same as Matlab on lower end, fixed upper limit to 255

im=cv2.imread('2.jpg',1);

(y,x,clrs) = im.shape
img_hou = np.copy(im[0:y/2, 0:x]) # Cropping Image (matlab line 16)

if plt_flag == 1:
    fig1 = plt.figure()
    plt.title('Initial Image')
    plt.imshow(im)
if plt_flag == 2:
    cv2.imshow('Initial Image',im)
    

# Alter contrast, brightness

if lim_type == 1:
    lims = stretchlim(im)
if lim_type == 2:
    lims = stretchlim2(im)

img2 = np.copy(imadjust(im,lims))
lims_hou = stretchlim(img_hou)
img2_hou = np.copy(imadjust(img_hou,lims_hou))

if plt_flag == 1:
    fig2 = plt.figure()
    plt.title('imadjust')
    plt.imshow(img2,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('imadjust',img2)

# Remove Background

img_remove_hou = np.copy(back_ground_remove(img2_hou))
img_remove = np.copy(back_ground_remove(img2))

if plt_flag == 1:
    fig3 = plt.figure()
    plt.title('Remove Background')
    plt.imshow(img_remove_hou,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('Remove Background',img_remove)

img_seg_hou = image_segmentation(img_remove_hou)
img_seg = image_segmentation_length(img_remove)

if plt_flag == 1:
    fig4 = plt.figure()
    plt.title('Image Segmentation')
    plt.imshow(img_seg_hou,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('Image Segmentation',img_seg_hou)

# Edge detection
img_edge = cv2.Canny(img_seg_hou,100,200)

if plt_flag == 1:
    fig5 = plt.figure()
    plt.title('Edge Detection')
    plt.imshow(img_edge,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('Edge Detection',img_edge)


# img_edge = edge(img_seg_hou,'canny');
# Nowhere else in the matlab code was this used, commented out

color1 =['r','g','b','c','m','y']
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

if plt_flag == 1:
    fig5 = plt.figure()
    sbplt = fig5.add_subplot(111)
    sbplt.imshow(im)

#####################################
#################### CIRCLE DETECTION
circles = cv2.HoughCircles(img_seg_hou, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

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

if(plt_flag == 1):
    fig6 = plt.figure()
    plt.imshow(img_seg_hou,cmap='Greys_r')
    for n in range(n_wr):
        circ_plt = plt.Circle((center_x2[n],center_y2[n]), 
                              radius[n], color=color1[1], fill=False)
        cntr_plt = plt.Circle((center_x2[n],center_y2[n]), 
                             1, color=color1[2])
        plt.gca().add_patch(circ_plt)
        plt.gca().add_patch(cntr_plt)
     
     
     
color1_l = np.fliplr([color1]);


#################### WRENCH REGION DETECTION
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

fig8 = plt.figure()
plt.imshow(im,cmap='Greys_r')
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
    
######################
#############   VOTING
    
    
#positions = np.zeros((3,mm))
#positions[0,:] = np.arange(mm)
#positions[1,:] = x_area[0:mm]
#positions[2,:] = x_length[0:mm:]
#
#e=40 # Pixels, not sure what this does.
#
##Voting goes here, not sure how it works or why.
#    
#if plt_flag == 1: 
#    plt.show()
#if plt_flag == 2:
#    cv2.waitKey(0)

