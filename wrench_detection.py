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
from scipy.cluster import vq

from imadjust import imadjust
from stretchlim import stretchlim, stretchlim2
from back_ground_remove import back_ground_remove
from image_segmentation import image_segmentation
from image_segmentation_length import image_segmentation_length
from kmeans import cluster_points, reevaluate_centers, find_centers

plt_flag = 1 # 0: Don't plot
             # 1: Plot using plt
             # 2: Plot using cv2.imshow
lim_type = 2 # 1: Same as Matlab (I think)
             # 2: Same as Matlab on lower end, fixed upper limit to 255
vot_type = 1 # 1: Tamer's voting algorithm
             # 2: Gaussian voting algorithm

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

img_seg_hou, img_gray_hou = image_segmentation(img_remove_hou)
img_seg = image_segmentation_length(img_remove)

if plt_flag == 1:
    fig4 = plt.figure()
    plt.title('Image Segmentation')
    plt.imshow(img_seg_hou,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('Image Segmentation',img_seg_hou)

# Edge detection
img_edge = cv2.Canny(img_seg_hou,30,100)

if plt_flag == 1:
    fig5 = plt.figure()
    plt.title('Edge Detection')
    plt.imshow(img_edge,cmap='Greys_r')
if plt_flag == 2:
    cv2.imshow('Edge Detection',img_edge)


# img_edge = edge(img_seg_hou,'canny');
# Nowhere else in the matlab code was this used, commented out

color1 =['r','g','b','c','m','y']
color2 = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255)]
radius_array =[16,17,18,21,25,28]
size_act=['19','18','15','14','13','12']

"""This comment block contains an attempt at recreating the manual circle
detection Tamer was using in Matlab. It is computationally expensive and
does not work well since python/opencv do not contain a good alternative
to the houghpeaks function in Matlab

kernelx = np.array([-0.5, 0, 0.5])
kernely = np.transpose(kernelx)

img_grad = img_edge.copy()

img_grad = cv2.filter2D(img_grad, 0, kernelx, -1)
img_grad = cv2.filter2D(img_grad, 0, kernely, -1);

rows, cols = np.where(img_grad == 0)
img_grad_len = len(rows)
sz = np.shape(img_grad)
accum_arry = np.zeros([sz[0],sz[1]])

radius = 20

for i in range(0,img_grad_len):
    x_c = cols[i]
    y_c = rows[i]
    x = x_c-radius * np.cos(img_grad[y_c,x_c])
    y = y_c-radius * np.sin(img_grad[y_c,x_c])

    x = round(x)
    y = round(y)
    if x <= sz[1] and x>0 and y <= sz[0] and y>0:
        accum_arry[y,x]=accum_arry[y,x]+1;
    
#if radius > 10:
#    num_peaks = 1
#else:
#    num_peaks = 5

#hihi = scipy.signal.find_peaks_cwt(accum_arry,10)

"""

if plt_flag == 1:
    fig5 = plt.figure()
    sbplt = fig5.add_subplot(111)
    sbplt.imshow(im)

#####################################
#################### CIRCLE DETECTION

circles = cv2.HoughCircles(img_gray_hou, cv2.cv.CV_HOUGH_GRADIENT,1,1,np.array([]),100,30,0,100)

center_x = circles[0,:,0] 
center_y = circles[0,:,1]
radius = circles[0,:,2]

sz = np.shape(img_gray_hou)
circs =  np.zeros([sz[0],sz[1]])

for i in range(0,len(center_x)):
    circs[round(center_y[i]),round(center_x[i])] = circs[round(center_y[i]),round(center_x[i])] + 1

# Apply KMeans
z = np.vstack((center_x,center_y))
z = np.transpose(z)
term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
ret, labels, centers = cv2.kmeans(z, 6, term_crit, 10, 0)

rad_kmean = np.zeros([6,1])
for i in range(0,6):
    locs = np.where(labels == i)
    rad = radius[locs[0]]
    rad_kmean[i] = np.mean(rad)
print "The wrench diameters I see are: ", np.transpose(rad_kmean)

id_r = len(center_y)

if(id_r>6):
    #Found more than 6 circles, n_wrench = 6
    n_wr = 6
else:
    n_wr = id_r

center_x2 = center_x[0:n_wr]
center_y2 = center_y[0:n_wr]
radius_2 = radius[0:n_wr]

if(plt_flag == 1):
    fig6 = plt.figure()
    plt.imshow(img_hou,cmap='Greys_r')
    for n in range(id_r):
        circ_plt = plt.Circle((center_x[n],center_y[n]), 
                              radius[n], color=color1[0], fill=False)
        cntr_plt = plt.Circle((center_x[n],center_y[n]), 
                             1, color=color1[2])
        plt.gca().add_patch(circ_plt)
        plt.gca().add_patch(cntr_plt)
if plt_flag == 2:
    img_hou_all = img_hou.copy()
    for n in range(id_r):
        cv2.circle(img_hou_all,(center_x[n],center_y[n]), radius[n], color2[0], 2, cv2.CV_AA)
    cv2.imshow('All Circles',img_hou_all)
     
center_x2 = centers[:,0]
center_y2 = centers[:,1]

if(plt_flag == 1):
    fig6 = plt.figure()
    plt.imshow(img_hou,cmap='Greys_r')
    for n in range(n_wr):
        circ_plt = plt.Circle((center_x2[n],center_y2[n]), 
                              rad_kmean[n], color=color1[n], fill=False)
        cntr_plt = plt.Circle((center_x2[n],center_y2[n]), 
                             1, color=color1[2])
        plt.gca().add_patch(circ_plt)
        plt.gca().add_patch(cntr_plt)
if plt_flag == 2:
    img_hou_kmeans = img_hou.copy()
    for n in range(n_wr):
        cv2.circle(img_hou_kmeans,(center_x2[n],center_y2[n]), rad_kmean[n], color2[n], 2, cv2.CV_AA)
    cv2.imshow('KMeans Circles',img_hou_kmeans)

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

if vot_type == 1:
    votings = np.zeros([2,n_wr])
    votings[0,:] = c_area2
    votings[1,:] = 50

    e = 40
    for i in range(0,n_wr):
        v_min = c_length2[i]-e
        v_max = c_length2[i]+e
        if (votings[0,i] >= v_min and votings[0,i] <= v_max):
            votings[1,i] = votings[1,i] + 30
    
    if (votings[1,n_wr-2] >= votings[1,n_wr-1] or votings[1,n_wr-2] >= votings[1,n_wr-3]):
        position = c_area2[n_wr-2]
    if (votings[1,n_wr-1] >= votings[1,n_wr-2] or votings[1,n_wr-1] >= votings[1,n_wr-3]):
        position = c_area2[n_wr-1]
    if (votings[1,n_wr-3] >= votings[1,n_wr-2] or votings[1,n_wr-3] >= votings[1,n_wr-1]):
        position = c_area2[n_wr-3]

    print "Correct wrench x-position is: ", position
#positions = np.zeros((3,mm))
#positions[0,:] = np.arange(mm)
#positions[1,:] = x_area[0:mm]
#positions[2,:] = x_length[0:mm:]
#
#e=40 # Pixels, not sure what this does.
#
##Voting goes here, not sure how it works or why.
#    






if plt_flag == 1: 
    plt.show()
if plt_flag == 2:
    cv2.waitKey(0)

