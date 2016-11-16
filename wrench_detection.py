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
#s=regionprops(img_seg,{'Centroid','BoundingBox','Area'});
# Implement Detect Length and Area:
#
#(contours, hierarchy) = cv2.findContours(img_seg,1,2)
#
#numobjs = len(contours)
#x = np.zeros(numobjs)
#y = np.zeros(numobjs)
#w = np.zeros(numobjs)
#h = np.zeros(numobjs)
#areas = np.zeros(numobjs)
#lengths = np.zeros(numobjs)
#
#for n,contour in enumerate(contours):
#    (x[n],y[n],w[n],h[n]) = cv2.boundingRect(contour)
#    
#    areas[n] = w[n]*h[n]
#    b_w = 1 # Bounding Box; needs implementation
#    lengths[n] = np.sqrt(w[n]**2 + h[n]**2)
#    # Need to implement lines 67 to 130, find area, length, centriod
#    # of rectangles, plot on original graph
#    
#if(len(lengths[lengths>0])<6):
#    mm= len(lengths[lengths>0])
#else:
#    mm = 6 # Not sure why this is the variable name, a lot of code
#            # goes into determining some of the parameters in this section
#
#
#x_length = np.zeros(mm)
#x_area = np.zeros(mm)
#y_area = np.zeros(mm)
#
#for n in range(mm):
#    cen = [x[n],y[n]]  # Need actual centroid location
#    bb = [x[n],y[n],x[n]+w[n],y[n]+h[n]] # Need actual bounding box
#    x_area[n] = x[n]*w[n] #Incorrect, placeholder
#    y_area[n] = y[n]*h[n] # Incorrect, placeholder
#    
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

