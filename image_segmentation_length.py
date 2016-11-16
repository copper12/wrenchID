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
    threshold = 3

    ret,final = cv2.threshold(img2, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(final.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(final.shape[:2], dtype="uint8") * 255
 
    # loop over the contours
    for c in cnts:
        area = cv2.contourArea(c)
        if area < threshold:
            cv2.drawContours(mask,[c], -1, 0, -1)
    final2 = cv2.bitwise_and(final, final, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    remove = cv2.morphologyEx(final2, cv2.MORPH_OPEN, kernel)
    remove = (255-remove)
    #remove = cv2.morphologyEx(remove, cv2.MORPH_OPEN, kernel)
    return remove
    
