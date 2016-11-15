# -*- coding: utf-8 -*-
"""
image_segmentation_length.py

Conversion of MATLAB code to python 2.7
Stefan Kraft, November 15, 2016
"""

import cv2
import numpy as np

def image_segmentation_length(img1):
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(img2,3)
    
    ret,final = cv2.threshold(img2, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    remove = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel)
#    remove = 
    return remove
    