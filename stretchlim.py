# -*- coding: utf-8 -*-
"""
imadjust.py
Conversion of MATLAB code to python 2.7
Jonathan Hodges, November 15, 2016
"""
import cv2
import numpy as np
import math

def stretchlim(img):
    # Determine size of image in pixels
    sz = np.shape(img)
    num_of_px = sz[0]*sz[1]
    # Determine one percent of total pixels (for use in image adjust code)
    one_perc = math.floor(num_of_px*0.01)
    lims = np.zeros((sz[2],2))
    # Compute lower/upper 1% threshold for each channel
    for i in range(0,sz[2]):
        hist,bins = np.histogram(img[:,:,i].ravel(),255,[0,255])
        val = 0; j = 0;
        while val < one_perc:
            val = val+hist[j]
            j = j +1
        lims[i,0] = j-2
        val = 0; j = 0;
        while val < one_perc:
            val = val+hist[254-j]
            j = j + 1
        lims[i,1] = 254-j+2
    return lims
