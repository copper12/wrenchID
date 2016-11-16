# -*- coding: utf-8 -*-
"""
back_ground_remove.py
Conversion of MATLAB code to python 2.7
Stefan Kraft, November 15, 2016
Modified Jonathan Hodges, November 15, 2016
"""
import numpy as np

def back_ground_remove(I):
    # Determine size of image in pixels
    sz = np.shape(I)
    # Initialize intensity array
    i = np.zeros((sz[2],1))
    # Initialize updated intensity matrix
    I3 = I.copy()
    # Loop through each channel of the image
    for j in range(0,sz[2]):
        # Caculate the intensity histogram of one channel
        hist,bins = np.histogram(I[:,:,j].ravel(),255,[0,255])
        I2 = I[:,:,j].copy()
        # Find the most common bin in the histogram
        i[j] = np.argmax(hist)
        # Fix overflow problems by setting values greater than the
        # modifier such that they will be maxed after addition
        I2[I2 > 255-i[j]*0.5] = 255-i[j]*0.5
        # Add the intensity modifier
        I2 = I2+0.5*i[j]
        # Update intensity matrix
        I3[:,:,j] = I2
    return I3
