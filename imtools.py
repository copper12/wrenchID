# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:05:05 2016

@author: skraft
"""

from PIL import Image
import numpy as np
import cv2

def imresize(im,sz):
    pil_im = Image.fromarray(uint8(im))
    
    return np.array(pil_im.resize(sz))
    

def histeq(im,nbr_bins=256):
    # Get Image Histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf/cdf[-1]
    
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape),cdf
    
def imadjust(im):
    # Function to return equilized image
    # similar to imadjust function in matlab
    # with stretchlim as the input argument
    
    imsplit = cv2.split(im)  
    imsplit[0] = cv2.equalizeHist(imsplit[0])
    imsplit[1] = cv2.equalizeHist(imsplit[1])
    imsplit[2] = cv2.equalizeHist(imsplit[2])
        
    
    immerge = cv2.merge((imsplit[0],imsplit[1],imsplit[2]))
    return immerge



def remove_background(im):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.BackgroundSubtractorMOG2()
    
    fgmask = fgbg.apply(im)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    imgresult = cv2.bitwise_and(im,im,mask=fgmask)
    
    return imgresult