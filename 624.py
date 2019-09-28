'''
-*- coding: utf-8 -*-
@Author  : LiZhichao
@Time    : 2019/6/24 8:48
@Software: PyCharm
@File    : 624.py
'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def dis(title,num,files):
    cv.namedWindow(title, 0)
    cv.resizeWindow(title, num * H, W)
    cv.imshow(title, np.hstack(files))
    cv.waitKey(0)
    cv.destroyAllWindows()

def display(num,files):
    for i in range(num):
        plt.figure("hist")
        arr = files[i].flatten()
        plt.subplot(1,2,i+1)
        n, bins, patches = plt.hist(arr, bins=256, density=1,edgecolor='None',
                                    facecolor='red')
    plt.show()
    plt.show()

#灰度上移
def huidushangyi(img1):
    img2 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if int(img1[i,j]+50)>255:
                gray = 255
            else:
                gray = int(img1[i,j]+50)
            img2[i,j]=gray
    flies = [img1,img2]
    dis("1",2,files=flies)

#灰度对比度
def huiduduibidu(img1):
    img3 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if img1[i,j] > 160:
                gray = img1[i,j]*1.2
                if gray > 255:
                    gray = 255
            elif img1[i,j] < 127:
                gray = img1[i,j]*0.5
            else:
                gray = img1[i,j]
            img3[i,j]=gray
    flies = [img1,img3]
    dis("2",2,files=flies)

#灰度反转
def huidufanzhuan(img1):
    img4 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            gray = 255 - img1[i,j]
            img4[i,j]=gray
    flies = [img1,img4]
    dis("3",2,files=flies)

#伽马
def gama(img1):
    img5 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            gray = 3 * pow(img1[i,j],0.8)
            img5[i,j]=gray
    flies = [img1,img5]
    dis("4",2,files=flies)

#二值化
def erzhihua(img1):
    img6 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if img1[i,j]>127:
                gray = 255
            else:
                gray = 0
            img6[i,j]=gray
    flies = [img1,img6]
    dis("5",2,files=flies)

#直方图均衡
def own(img):
    Newimg = np.zeros((H, W), np.uint8)  # 二维数组
    Hist = np.zeros(256, np.int)  # Pixel sum一维均衡化前
    EqHist = np.zeros(256, np.int)  # Equal Pixel均衡化后
    HistP = np.zeros(256, np.float)  # 像素概率
    HistPSUM = np.zeros(256, np.float)  # 像素概率和
    Pixelsum = H * W
    for i in range(H):
        for j in range(W):
            # Every Gray Pixel sum
            Hist[img[i,j]]+=1#像素点出现的次数

    for i in range(256):
        HistP[i] =Hist[i]/Pixelsum#像素点出现的概率

    for i in range(1,256):
        HistPSUM[i] =HistP[i]+HistPSUM[i-1]#归一化

    for i in range(256):
        EqHist[i] =HistPSUM[i]*255#均衡化后灰度值

    for i in range(H):
        for j in range(W):
            # Set New pixels Gray
            Newimg[i,j]= EqHist[img[i,j]]
    # plt.figure("hist")
    # arr = img.flatten()
    # n, bins, patches = plt.hist(arr, bins=256, density=1, edgecolor='None',
    #                             facecolor='red')
    # plt.show()
    #
    # plt.figure("hist")
    # arr = Newimg.flatten()
    # n, bins, patches = plt.hist(arr, bins=256, density=1, edgecolor='None',
    #                             facecolor='red')
    # plt.show()

    files = [img,Newimg]
    dis("7",2,files)
    display(2,files)
    return Newimg

#3x3中值滤波
def cross3x3(img):
    curimg = np.zeros((H,W),np.uint8)
    newimg = np.zeros((H,W),np.uint8)
    for i in range(1,H-1):
        for j in range(1,W-1):
            t00 = img[i-1, j - 1]
            t01 = img[i-1, j]
            t02 = img[i-1, j + 1]
            t10 = img[i, j-1]
            t11 = img[i, j]
            t12 = img[i, j+1]
            t20 = img[i + 1, j-1]
            t21 = img[i + 1, j]
            t22 = img[i + 1, j+1]
            templ =[t00,t01,t02,t10,t11,t12,t20,t21,t22]
            templ.sort()
            curimg[i,j]=templ[4]
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if curimg[i,j] != 0:
                newimg[i,j] = curimg[i,j]
            else:
                newimg[i,j] = img[i,j]
    files = [img,curimg,newimg]
    dis('3x3', 3, files)

if __name__ == '__main__':
    img = cv.imread("cell.jpg")
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H = img1.shape[0]
    W = img1.shape[1]
    pixel = H * W
    print(img.shape)
    B, G, R = cv.split(img)

    # huidushangyi(img1)
    # huiduduibidu(img1)
    # huidufanzhuan(img1)
    # gama(img1)
    # erzhihua(img1)
    # imgh = own(img1)
    # cross3x3(imgh)
    imga = cv.medianBlur(img1,3)
    imga1 = cv.medianBlur(img1,5)
    imga2 = cv.medianBlur(img1,7)
    imgb = cv.GaussianBlur(img1,(7,7),0)
    imgc = cv.blur(img1,(5,5))
    imgd = cv.bilateralFilter(img1,127,75,75)
    cv.imshow("imga",imga)
    cv.imshow("imga1",imga1)
    cv.imshow("imga2",imga2)
    cv.imshow("imgb",imgb)
    cv.imshow("imgc",imgc)
    cv.imshow("imgd",imgd)

    cv.waitKey(0)
    cv.destroyAllWindows()

