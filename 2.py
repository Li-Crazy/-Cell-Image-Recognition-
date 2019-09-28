import numpy as np
import cv2 as cv
import copy
from skimage import exposure
 
a = cv.imread('cell.jpg')
 
kernel = np.ones((3,3),np.uint8)  
w = int(a.shape[0]/5)
h = int(a.shape[1]/5)
a = cv.resize(a,(h,w))
ry = a.copy()
rry = a.copy()
#a[a>0.3*255] = 255
#a[a<=0.3*255] = 0
 
parameter=10.2
gamma_img = exposure.adjust_gamma(a, parameter)
gamma_img = cv.cvtColor(gamma_img, cv.COLOR_BGR2GRAY)
gamma_img[gamma_img<0.3*255] = 0
#gamma_img = cv.medianBlur(gamma_img ,3)
#gamma_img = cv.medianBlur(gamma_img ,3)
gamma_img = cv.erode(gamma_img,kernel,iterations = 1)
gamma_img = cv.dilate(gamma_img,kernel,iterations = 2)
 
cv.imshow('bccccccccc',gamma_img)
cv.waitKey(0)
cv.destroyAllWindows()
 
#gamma_img = cv.Canny(gamma_img,50,150)
#gamma_img[gamma_img<=0.6*255] = 0
 
#gamma_img[gamma_img<0.3*255] = 0
#gamma_img[gamma_img>0.5*255] = 0
temp = np.ones(a.shape,np.uint8)*0
#gamma_img[gamma_img>=0.3*255 and gamma_img<0.9*255] = 255
hdddd = cv.findContours(gamma_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  
contours = hdddd[1]
dddd = cv.drawContours(temp,contours,-1,(255,255,255),5)
 
contour_index = []
for i in contours:
    
    print(i.shape[0])
    contour_index.append(i)
#max_contour_index = contour_index.index(max(contour_index))
 
#aaaaaaaaaaaaaa = contours[max_contour_index]
contour_index = np.concatenate([en for en in contour_index],axis=0)
 
tt = np.ones(a.shape,np.uint8);
(x, y), radius = cv.minEnclosingCircle(contour_index)
(x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
print(x, y, radius)
cv.circle(tt, (x, y), radius, (0, 0, 255), 2)
cv.imshow('tt',tt)
cv.waitKey(0)
cv.destroyAllWindows()
 
cv.imshow('bccccccccc',gamma_img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('bccccccccc',temp)
cv.waitKey(0)
cv.destroyAllWindows()
 
 
 
#cv.imshow('bccccccccc',tt)
#cv.waitKey(0)
#cv.destroyAllWindows()
for i in range(w):
    for j in range(h):
        if np.sqrt(np.square(i-y)+np.square(j-x))>radius:
            ry[i,j]=0
cv.imshow('ry',ry)   
cv.waitKey(0)
cv.destroyAllWindows()
#-------------------以上完成切割操作，ry为切割后的圆盘内部图像------------------------
 
 
ry = exposure.adjust_gamma(ry, 10.2)
ry_g = cv.cvtColor(ry,cv.COLOR_BGR2GRAY)
ry_g[ry_g>200] = 0
ry_g[ry_g<50] = 0
ry_g = cv.medianBlur(ry_g ,3)
ry_g = cv.dilate(ry_g,kernel,iterations = 2)
#ry_canny = cv.Canny(ry,0,200)
#cv.imshow('ry_canny',ry_canny)   
#cv.waitKey(0)
#cv.destroyAllWindows()
 
cv.imshow('rry',ry_g)   
cv.waitKey(0)
cv.destroyAllWindows()
 
 
ry_hd = cv.findContours(ry_g,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  
ry_contours = ry_hd[1]
#dddd = cv.drawContours(temp,ry_contours,-1,(255,255,255),2)
#cv.imshow('ry_canny',temp)   
cv.waitKey(0)
cv.destroyAllWindows()
cclist = []
for i in ry_contours:
    
    print(i.shape[0])
    if i.shape[0]<50 and i.shape[0]>3:
        cclist.append(i)
        
 
 
temp1 = np.ones(a.shape,np.uint8)*0
dddd11 = cv.drawContours(rry,cclist,-1,(255,0,255),2)
cv.imshow('ry_canny',dddd11)   
cv.waitKey(0)
cv.destroyAllWindows()
 
 
 
 
 
#kernel = np.ones((3,3),np.uint8)  
#gamma_img = cv.dilate(gamma_img,kernel,iterations = 2)
 
circles = cv.HoughCircles(gamma_img,cv.HOUGH_GRADIENT,1,1,
                            param1=100,param2=10,minRadius=0,maxRadius=10)
 
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(a,(i[0],i[1]),i[2],(255,0,0),2)
    # draw the center of the circle
    cv.circle(a,(i[0],i[1]),2,(255,0,0),3)
    
cv.imshow('detected circles',a)
cv.waitKey(0)
cv.destroyAllWindows()
#w = int(a.shape[0]/5)
#h = int(a.shape[1]/5)
#
#a = cv.resize(a,(h,w))
#b = a.copy()
#f = a.copy()
#a = cv.medianBlur(a,9)
#a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
#a = cv.Canny(a,0,255)
#kernel = np.ones((6,6),np.uint8) 
##a = cv.erode(a,kernel,iterations = 20)
#a = cv.dilate(a,kernel,iterations = 20)
#cv.imshow('ya',a)
#cv.waitKey(0)
#cv.destroyAllWindows()
#
#
#
#temp = np.ones(a.shape,np.uint8)*0
#
#h = cv.findContours(a,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
#contours = h[1]
#dddd = cv.drawContours(temp,contours,-1,(255,255,255),3)
#
#
#cv.imshow('dddd111111',dddd)
#cv.waitKey(0)
#cv.destroyAllWindows()
##
#tt = np.ones(dddd.shape,np.uint8);
#(x, y), radius = cv.minEnclosingCircle(contours[0])
#(x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
#print(x, y, radius)
#cv.circle(tt, (x, y), radius, (0, 0, 255), 2)
#cv.imshow('bccccccccc',tt)
#cv.waitKey(0)
#cv.destroyAllWindows()