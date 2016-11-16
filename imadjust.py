# -*- coding: utf-8 -*-
"""
imadjust.py
Conversion of MATLAB code to python 2.7
Jonathan Hodges, November 15, 2016
"""
import numpy as np

def imadjust(img,lims):
    sz = np.shape(img)
    for i in range(0,sz[2]):
        img[:,:,i] = (img[:,:,i]-lims[i,0])/(lims[i,1]-lims[i,0])*255
    return img
