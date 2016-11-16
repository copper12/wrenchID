# -*- coding: utf-8 -*-
"""
imadjust.py
Conversion of MATLAB code to python 2.7
Jonathan Hodges, November 15, 2016
"""
import numpy as np

def imadjust(img,lims):
    img2 = np.copy(img)
    sz = np.shape(img2)
    for i in range(0,sz[2]):
        I2 = img2[:,:,i]
        I2[I2 > lims[i,1]] = lims[i,1]
        I2[I2 < lims[i,0]] = lims[i,0]
        img2[:,:,i] = (I2-lims[i,0])/(lims[i,1]-lims[i,0])*255
    return img2
