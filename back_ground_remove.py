# -*- coding: utf-8 -*-
"""
back_ground_remove.py

Conversion of MATLAB code to python 2.7
Stefan Kraft, November 15, 2016
"""
import numpy as np

def back_ground_remove(I):
    
    
    (counts,x) = np.histogram(I[0],bins=256)    
    i0 = np.argmax(counts)
    (counts,x) = np.histogram(I[1],bins=256)    
    i1 = np.argmax(counts)
    (counts,x) = np.histogram(I[2],bins=256)    
    i2 = np.argmax(counts)

#      % i1=200;
#      % i2=250;
#      % i3=185;
    
    r2=I[:,:,0]+0.5*i0
    g2=I[:,:,1]+0.5*i1
    b2=I[:,:,2]+0.5*i2
    
    I2=I
    I2[:,:,0]=r2
    I2[:,:,1]=g2
    I2[:,:,2]=b2
#    i=[i1 i2 i3];
    
    return I2