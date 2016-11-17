# -*- coding: utf-8 -*-
"""
wrench_detection.py

Conversion of MATLAB code to python 2.7
Background Removal Algorithm - Tamer Attia
Initial conversion of MATLAB code - Stefan Kraft, 11/15/2016
Finished conversion of MATLAB code, improved detection algorithms,
implement gaussian based voting algorithm - Jonathan Hodges, 11/17/2016
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.cluster import vq
import scipy.stats

from imadjust import imadjust
from stretchlim import stretchlim, stretchlim2
from back_ground_remove import back_ground_remove
from image_segmentation import image_segmentation
from image_segmentation_length import image_segmentation_length

########################################################################
############################# User Inputs ##############################
########################################################################

plt_flag = 1 # 0: Don't plot
             # 1: Plot using plt
             # 2: Plot using cv2.imshow
plt_qty = 1  # 0: Only plot final product
             # 1: Verbose plotting
lim_type = 1 # 1: Same as Matlab (I think)
             # 2: Same as Matlab on lower end, fixed upper limit to 255
vot_type = 2 # 1: Tamer's voting algorithm (Not fully implemented)
             # 2: Gaussian voting algorithm

n_wr = 6 # Number of wrenches
im=cv2.imread('2.jpg',1); # Image to read

# Tweaking parameters - Need to be adjusted if we are at different dist
area_min_thresh = 3000 # Minimum contour area to be a wrench
max_circ_diam = 100 # Maximum circle diameter considered
canny_param = [100, 30] # Canny edge detection thresholds
p2crop = 2 # Portion of image to crop for circle detection

d_mu = 21 # Diameter of correct wrench in pixels
d_sig = d_mu*0.1 # Uncertainty in diameter
l_mu = 375 # Length of correct wrench in pixels 
l_sig = l_mu*0.1 # Uncertainty in length
a_mu = 11500 # Area of correct wrench in pixels
a_sig = a_mu*0.1 # Uncertainty in area
vote_wt = [0.33,0.33,0.33] # Weight of each criteria in voting (d,l,a)

########################################################################
######################## Image Pre-Processing ##########################
########################################################################
# This section takes an RGB image as an input and prepares it for
# detection by removing the background.

(y,x,clrs) = im.shape # Get size of image
img_hou = np.copy(im[0:y/p2crop, 0:x]) # Crop image for circle detection

# Determine ideal limits for brightness/contrast adjustment
if lim_type == 1:
    lims = stretchlim(im)
    lims_hou = stretchlim(img_hou)
if lim_type == 2:
    lims = stretchlim2(im)
    lims_hou = stretchlim2(img_hou)

# Adjust the brightness/contrast of the RGB image based on limits
img2 = np.copy(imadjust(im,lims))
img2_hou = np.copy(imadjust(img_hou,lims_hou))

# Remove Background from adjusted brightness/contrast image
img_remove_hou = np.copy(back_ground_remove(img2_hou))
img_remove = np.copy(back_ground_remove(img2))

# Convert the image to binary
img_seg_hou, img_gray_hou = image_segmentation(img_remove_hou)
img_seg, img_gray = image_segmentation_length(img_remove)

# Edge detection
# NOTE: This is not actually in use. cv2.HoughCircles uses canny edge
#       detection internally, so passing an edge image to it does not
#       nice things.
img_edge = cv2.Canny(img_seg_hou,canny_param[0],canny_param[1])

########################################################################
########################### Circle Detection ###########################
########################################################################
# This section takes the pre-processed images and determines for each
# wrench:
# 1. diameter of the circle
# 2. vertical length of the wrench
# 3. area of the wrench

# Find all circles in the image. Last 4 parameters are:
# 1. Canny edge detection uppper threshold
# 2. Canny edge detection lower threshold
# 3. Minimum circle radius
# 4. Maximum circle radius
circles = cv2.HoughCircles(img_gray_hou, cv2.cv.CV_HOUGH_GRADIENT,1,1,
            np.array([]),canny_param[0],canny_param[1],0,max_circ_diam)

# Store center coordinates and radii in a more readable format
center_x = circles[0,:,0]
center_y = circles[0,:,1]
radius = circles[0,:,2]
id_r = len(center_y)

# Quantize the circle centers into n_wr (number of wrenches) groups

# Establish matrix of features to use for quanitzation
z = np.transpose(np.vstack((center_x,center_y)))

# Run K-means to determine centers and to which group each point
# belongs.
term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
ret, labels, centers = cv2.kmeans(z, n_wr, term_crit, 10, 0)

# Find average radius within each K-means group
rad_kmean = np.zeros([6,1])
for i in range(0,6):
    locs = np.where(labels == i)
    rad = radius[locs[0]]
    rad_kmean[i] = np.mean(rad)

# Store center coordinates from K-means in more readable format
circs = np.zeros([6,3])
circs[:,0:2] = centers
circs[:,2:] = rad_kmean

# Sort circles by x-axis
circs = circs[circs[:,0].argsort()]

img_hou_all = img_hou.copy()
for n in range(id_r):
    cv2.circle(img_hou_all,(center_x[n],center_y[n]), radius[n],
        (0,0,244), 2, cv2.CV_AA)

########################################################################
########################### Length Detection ###########################
########################################################################
# Create a copy of the grayscale image
img_gray_con = img_gray.copy()

# Find Contours
cnt, hie = cv2.findContours(img_seg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Remove contours which are too small to be a wrench based on area
cnt2 = []; hie2 = np.zeros([6,12,4]); ct = 0
cen2 = np.zeros([6,2]); len2 = np.zeros([6,2])
area2 = np.zeros([6,1])

for c in range(0,len(cnt)):
    area = cv2.contourArea(cnt[c])
    if area > area_min_thresh:
        area2[ct] = area
        cnt2.append(cnt[c])
        hie2[ct,:,:] = hie
        M = cv2.moments(cnt[c])
        cen2[ct,0] = int(M["m10"] / M["m00"])
        cen2[ct,1] = int(M["m01"] / M["m00"])
        (hi1,hi2,len2[ct,0],len2[ct,1]) = cv2.boundingRect(cnt[c])
        ct = ct+1

# Store all relevant features in a single matrix
params = np.zeros([6,6])
params[:,0:2] = cen2
params[:,2:4] = len2
params[:,4:] = area2.reshape((6,1))

# Sort contour list by x position
ind = np.argsort(params[:,0])
cnt3 = []
for i in range(0,n_wr):
    cnt3.append(cnt2[ind[i]])

# Sort feature matrix by x position of centroid
params = params[params[:,0].argsort()]

# Store circle diameters in feature matrix
params[:,5:] = circs[:,2:]

# Provide feedback on wrench locations and diameters
print "Wrenches: (column, row, width, height, area, diameter)"
print params

########################################################################
################################ Voting ################################
########################################################################

# The voting algorithm in vot_type == 1 was tailored to the specific
# image and wrench configuration in the simulation. Its use is not
# recommended.
#
# The voting algorithm in vot_type == 2 calculates the probability each
# wrench is the correct one using a Gaussian distribution for each of
# the three parameters.

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
    
    if (votings[1,n_wr-2] >= votings[1,n_wr-1] or votings[1,n_wr-2] 
            >= votings[1,n_wr-3]):
        position = c_area2[n_wr-2]
    if (votings[1,n_wr-1] >= votings[1,n_wr-2] or votings[1,n_wr-1] 
            >= votings[1,n_wr-3]):
        position = c_area2[n_wr-1]
    if (votings[1,n_wr-3] >= votings[1,n_wr-2] or votings[1,n_wr-3] 
            >= votings[1,n_wr-1]):
        position = c_area2[n_wr-3]

    print "Correct wrench x-position is: ", position

if vot_type == 2:
    votes = np.zeros([n_wr,3])
    votes_ideal = np.zeros([1,3])
    # Store the maximum possible probability for each parameter based
    # on the Gaussian distribution
    votes_ideal[0,0] = scipy.stats.norm(d_mu, d_sig).pdf(d_mu)
    votes_ideal[0,1] = scipy.stats.norm(l_mu, l_sig).pdf(l_mu)
    votes_ideal[0,2] = scipy.stats.norm(a_mu, a_sig).pdf(a_mu)
    # Compute the probability for each wrench using each parameter
    # based on the Gaussian distribution
    for i in range(0,n_wr):
        votes[i,0] = scipy.stats.norm(d_mu, d_sig).pdf(params[i,5])
        votes[i,1] = scipy.stats.norm(l_mu, l_sig).pdf(params[i,3])
        votes[i,2] = scipy.stats.norm(a_mu, a_sig).pdf(params[i,4])
    # Scale the probabilities based on the maximum possible for each
    # parameter
    votes = votes/votes_ideal
    vote_result = np.zeros([n_wr,1])
    # Sum the probabilities based on the weight values for each parameter
    vote_result = np.dot(votes,vote_wt)
    print "Vote results: ", vote_result
    ind = vote_result.argsort()
    print "The most likely wrench is (L = 0, R = 5): ", ind[n_wr-1]

    # Visualize the probabilities
    img_kmeans = im.copy()
    for n in range(n_wr):
        c = int(round(vote_result[n]*255))
        cv2.circle(img_kmeans,(int(circs[n,0]),int(circs[n,1])), 
            int(circs[n,2]), (0,c,255-c), 2, cv2.CV_AA)
        cv2.drawContours(img_kmeans, cnt3[n], -1, (0,c,255-c), 3)

    # Visualize the best match
    img_id = im.copy()
    n = ind[n_wr-1]
    cv2.circle(img_id,(int(circs[n,0]),int(circs[n,1])), 
        int(circs[n,2]), (0,255,0), 2, cv2.CV_AA)
    cv2.drawContours(img_id, cnt3[n], -1, (0,255,0), 3)

########################################################################
############################### Plotting ###############################
########################################################################

if plt_qty == 1:
    if plt_flag == 1:
        fig1 = plt.figure(); plt.title('Initial Image'); plt.imshow(im)
        fig2 = plt.figure(); plt.title('Adjusted Image');
        plt.imshow(img2,cmap='Greys_r')
        fig3 = plt.figure(); plt.title('Remove Background')
        plt.imshow(img_remove_hou,cmap='Greys_r')
        fig4 = plt.figure(); plt.title('Image Segmentation');
        plt.imshow(img_seg_hou,cmap='Greys_r')
        fig5 = plt.figure(); plt.title('Edge Detection');
        plt.imshow(img_edge,cmap='Greys_r')
        fig6 = plt.figure(); plt.title('All Circles');
        plt.imshow(img_hou_all)
    if plt_flag == 2:
        cv2.imshow('Initial Image',im)
        cv2.imshow('Brightness Adusted Image',img2)
        cv2.imshow('Remove Background',img_remove)
        cv2.imshow('Image Segmentation',img_seg_hou)
        cv2.imshow('Edge Detection',img_edge)
        cv2.imshow('All Circles',img_hou_all)
if plt_qty >= 0:
    if(plt_flag == 1):
        fig7 = plt.figure();
        plt.title('Probability (Green good, Blue bad)');
        plt.imshow(img_kmeans)
        fig8 = plt.figure(); plt.title('Most Likely Candidate');
        plt.imshow(img_id)
        plt.show()
    if plt_flag == 2:
        cv2.imshow('Probability (Green good, Red bad)',img_kmeans)
        cv2.imshow('Most Likely Candidate',img_id)
        cv2.waitKey(0)

