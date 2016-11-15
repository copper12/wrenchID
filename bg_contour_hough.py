#!/usr/bin/python

import cv2
#import cv2 as cv2.cv
import numpy as np
import sys
import matplotlib.pyplot as plt

#Read image
img = cv2.imread('/home/trec/Desktop/Detect/data/wrench_images/wrench.png',1)

# background removal
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
	
histB,binsB = np.histogram(b.ravel(),256,[0,256])
histG,binsG = np.histogram(g.ravel(),256,[0,256])
histR,binsR = np.histogram(r.ravel(),256,[0,256])
	
countB, _, _ = plt.hist(histB,binsB)
mB =  np.max(countB)
countG, _, _ = plt.hist(histG,binsG)
mG = np.max(countG)
countR, _, _ = plt.hist(histR,binsR)
mR = np.max(countR)

b1 = b + mB
g1 = g + mG
r1 = r + mR
	
img1 = img.copy()
img1[:,:,0] = b1
img1[:,:,1] = g1
img1[:,:,2] = r1
#cv2.imshow('bg',img1)
#cv2.waitKey(0)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img1,105,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2) # contour detection
scale_factor = 1
L = []
A = []
R = []
xL = []
xA = []
xR = []
yR = []
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt) 
	if h > 200  and h < 500: # constraint on the height of contours to eliminate other contours
		epsilon = 0.1*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		area = cv2.contourArea(cnt)
		A.append(area)

		cv2.drawContours(img, cnt, -1, (0,255,0), 3)
		L.append(h)
		xL.append(x + w/2)
		crop = img1[y-20:y+h/5,x:x+w] #crop image for circle detection
		cv2.rectangle(img,(x,y),(x+w,y+h),(100,255,0),2)
#                crop = cv2.resize(crop,(0,0),fx=scale_factor,fy=scale_factor)
		crop1 = cv2.medianBlur(crop,5)

		circles = cv2.HoughCircles(crop1, cv2.cv.CV_HOUGH_GRADIENT,1,10,np.array([]),100,38,10,40)
		if circles != None:
			a,b,c = circles.shape
			for i in range(b):
				cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), circles[0][i][2], (255,0,0), 1, cv2.CV_AA)
				cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), 2, (255,100,0), 1, cv2.CV_AA)
				x0 = int(circles[0][i][0]/scale_factor)
				y0 = int(circles[0][i][1]/scale_factor)
			# get the size of the circles
				R.append(circles[0][i][2]/scale_factor)
				xR.append(x0+x)
				yR.append(y0+y-20)
#					print circles[0][i][2]/scale_factor
				cv2.circle(img,(x0+x, y0+y-20), int(circles[0][i][2]/scale_factor), (255,0,0), 2, cv2.CV_AA)
				cv2.circle(img,(x0+x, y0+y-20), 2, (255,100,0), 1, cv2.CV_AA)
#		cv2.imshow('preview',img)
#		cv2.waitKey(0)

# sort wrenches by area, length and radius and select 3rd smallest one based on weights
A = np.array(A)
L = np.array(L)
R = np.array(R)
xL = np.array(xL)
xR = np.array(xR)
yR = np.array(yR)

sA = np.sort(A)
sL = np.sort(L)
sR = np.sort(R)

indexA = np.argsort(A)
indexR = np.argsort(R)
indexL = np.argsort(L)

x_L = np.zeros((np.size(xL,0)))
x_R = np.zeros((np.size(xR,0)))

for i in range(np.size(indexR)):
	x_R[i] = xR[indexR[i]]	
for i in range(np.size(indexL)):
	x_L[i] = xL[indexL[i]]

vote = np.zeros((np.size(x_L,0),1))
vote[indexL[2]] = vote[indexL[2]] + 0.5
vote[indexA[2]] = vote[indexA[2]] + 0.3
vote[indexR[2]] = vote[indexR[2]] + 0.2

index = np.argmax(vote)
wrench = np.array([xR[index],yR[index]])
cv2.imshow('detection',img)
cv2.imwrite('wDetection1.png',img)
#if wrench.size:
#	print wrench
#else:
#	print "Not Found"
cv2.waitKey(0)

