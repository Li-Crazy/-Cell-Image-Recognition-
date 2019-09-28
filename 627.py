'''
-*- coding: utf-8 -*-
@Author  : LiZhichao
@Time    : 2019/6/24 8:48
@Software: PyCharm
@File    : 624.py
'''
from tkinter import *
from tkinter import filedialog
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog
import os


def dis(title,num,files):
    cv.namedWindow(title, 0)
    cv.resizeWindow(title, num * H, W)
    cv.imshow(title, np.hstack(files))
    cv.waitKey(0)
    cv.destroyAllWindows()

def display(num,titles,files):
    for i in range(num):
        plt.figure("hist")
        arr = files[i].flatten()
        plt.subplot(1,2,i+1)
        plt.title(titles[i])
        n, bins, patches = plt.hist(arr, bins=256, density=1,edgecolor='None',
                                    facecolor='red')
    plt.show()

def displays(r,c,titles,files):
    for i in range(len(files)):
        plt.subplot(r,c,i+1)
        plt.imshow(files[i], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(titles[i])
    plt.show()

def grey(img):
    # 获取当前图片的信息
    imgInfo = img.shape
    heigh = imgInfo[0]
    width = imgInfo[1]
    # dst 一般是新建值，目标图片
    dst = np.zeros((heigh, width), np.uint8)
    for i in range(0, heigh):
        for j in range(0, width):
            gray = 0.114 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.299 * img[i, j, 2]
            dst[i, j] = np.uint8(gray)
    return dst

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
    dis("huidushangyi",2,files=flies)

#灰度下移
def huiduxiayi(img1):
    img2 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if int(img1[i,j]-50)< 0:
                gray = 0
            else:
                gray = int(img1[i,j]-50)
            img2[i,j]=gray
    flies = [img1,img2]
    dis("huiduxiayi",2,files=flies)

#灰度对比度
def huiduduibidu(img1):
    img3 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if img1[i,j] > 127:
                gray = img1[i,j]*1.2
                if gray > 255:
                    gray = 255
            elif img1[i,j] < 70:
                gray = img1[i,j]*0.5
            else:
                gray = img1[i,j]
            img3[i,j]=gray
    flies = [img1,img3]
    dis("huiduduibidu",2,files=flies)

#灰度反转
def huidufanzhuan(img1):
    img4 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            gray = 255 - img1[i,j]
            img4[i,j]=gray
    flies = [img1,img4]
    dis("huidufanzhuan",2,files=flies)
    return img4

#伽马
def gama(img1):
    img5 = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            gray = 3 * pow(img1[i,j],0.8)
            img5[i,j]=gray
    flies = [img1,img5]
    dis("gama",2,files=flies)

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
    dis("erzhihua",2,files=flies)

#直方图均衡
def Hist(img):
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

    files = [img,Newimg]
    titles = ["first","last"]
    displays(1,2,titles,files)
    display(2,titles,files)
    return Newimg

#直方图规定化
def cumFre(src):
    # get image size
    rows, cols = src.shape
    # get image histogram like (hist,bins = np.histogram(img.flatten(), 256, [0, 255]))
    hist = cv.calcHist([src], [0], None, [256], [0, 256])
    # get image hist_add is formula si
    cumHist = np.cumsum(hist)
    # Calculation of cumulative frequency of images
    cumf = cumHist / (rows*cols)
    return cumf

def histMatching(oriImage, refImage):
    oriCumHist = cumFre(oriImage)     #
    refCumHist = cumFre(refImage)     #
    lut = np.ones(256, dtype = np.uint8) * (256-1) #new search sheet
    start = 0
    for i in range(256-1):
        temp = (refCumHist[i+1] - refCumHist[i]) / 2.0 + refCumHist[i]
        for j in range(start, 256):
            if oriCumHist[j] <= temp:
                lut[j] = i
            else:
                start = j
                break

    dst = cv.LUT(oriImage, lut)
    return dst

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
    return newimg

def Robert(H,W,img):
    imgR = np.zeros((H,W),np.uint8)
    for i in range(0, H - 1):
        for j in range(0, W - 1):
            a00 = np.int16(img[i,j])
            a01 = np.int16(img[i,j+1])
            a10 = np.int16(img[i+1,j])
            a11 = np.int16(img[i+1,j+1])
            #u = np.sqrt((a00-a11)**2 + (a10-a01)**2)
            u = np.abs((a00-a11))+np.abs(a10-a01)
            if u>= 255:
                u = 255
            elif u<0:
                u = 0
            imgR[i,j] =np.uint8(u)
    files = [img,imgR]
    dis('Robert', 2, files)
    # return imgR

def Grad(H,W,img):
    imgG= np.zeros((H, W), np.uint8)
    for i in range(0, H - 1):
        for j in range(0, W - 1):
            a00 = np.int16(img[i, j])
            a01 = np.int16(img[i, j + 1])
            a10 = np.int16(img[i + 1, j])
            a11 = np.int16(img[i + 1, j + 1])
            u = np.abs(a00 - a10) + np.abs(a00 - a01)
            if u >= 255:
                u = 255
            elif u < 0:
                u = 0
            imgG[i, j] = np.uint8(u)
    files = [img,imgG]
    dis('Grad', 2, files)
    # return imgG

def Prewitt(H,W,img):
    imgpX = np.zeros((H, W), np.uint8)
    imgpY = np.zeros((H, W), np.uint8)
    imgpXY = np.zeros((H, W), np.uint8)
    imgpS = np.zeros((H, W), np.uint8)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a00 = img[i - 1, j - 1]
            a01 = img[i - 1, j]
            a02 = img[i - 1, j + 1]
            a10 = img[i, j - 1]
            a11 = img[i, j]
            a12 = img[i, j + 1]
            a20 = img[i + 1, j - 1]
            a21 = img[i + 1, j]
            a22 = img[i + 1, j + 1]
            ux = a20 * 1 + a10 * 1 + a00 * 1 + a02 * -1 + a12 * -1 + a22 * -1
            imgpX[i, j] = ux
            uy = a02 * 1 + a01 * 1 + a00 * 1 + a20 * -1 + a21 * -1 + a22 * -1
            imgpY[i, j] = uy
            imgpXY[i, j] = np.sqrt(ux * ux + uy * uy)
            imgpS[i, j] = np.abs(ux) + np.abs(uy)
    titles = ["img","imgpXY","imgpS"]
    files = [img, imgpXY, imgpS]
    displays(1,3,titles,files)

def Sobel(H,W,img):
    imgX = np.zeros((H,W),np.uint8)
    imgY = np.zeros((H,W),np.uint8)
    imgXandY = np.zeros((H,W),np.uint8)
    imgabS = np.zeros((H,W),np.uint8)
    for i in range(1,H-1) :
        for j in range (1,W-1):
            a00 = img[i-1, j-1]
            a01 = img[i-1, j]
            a02 = img[i-1, j+1]
            a10 = img[i,j-1]
            a11 = img[i,j]
            a12 = img[i, j+1]
            a20 = img[i+1, j-1]
            a21 = img[i+1, j]
            a22 = img[i+1, j+1]
            ux = a20 * 1 + a10 * 2 + a00 * 1  + a02 * -1 + a12 * -2 + a22 *-1
            imgX[i,j] = ux
            uy = a02 * 1 + a01 * 2 + a00 * 1+ a20 * -1 + a21 * -2 + a22 * -1
            imgY[i,j] = uy
            imgXandY[i,j] = np.sqrt(ux* ux+uy *uy)
            imgabS[i,j]  = np.abs(ux) + np.abs(uy)
    titles = ["img","imgXandY","imgabS"]
    files = [img,imgXandY,imgabS]
    displays(1,3,titles,files)

def Synthetic(H,W,img):
    # H6=np.mat([[0,-1,0],        H9 =[[-1,-1,-1]
    #           [-1,5,-1],            [-1, 9 ,-1]
    #           [0,-1,0]])            [-1,-1,-1]]
    imgXYH1 = np.zeros((H, W), np.uint8)
    imgXYH2 = np.zeros((H, W), np.uint8)
    imgXYH3 = np.zeros((H, W), np.uint8)
    imgXYH4 = np.zeros((H, W), np.uint8)
    for i in range(1, H - 2):
        for j in range(1, W - 2):
            a00 = np.int16(img[i - 1, j - 1])
            a01 = np.int16(img[i - 1, j])
            a02 = np.int16(img[i - 1, j + 1])
            a10 = np.int16(img[i, j - 1])
            a11 = np.int16(img[i, j])
            a12 = np.int16(img[i, j + 1])
            a20 = np.int16(img[i + 1, j - 1])
            a21 = np.int16(img[i + 1, j])
            a22 = np.int16(img[i + 1, j + 1])
            h1 = -a10 - a12 - a01 - a02 +5* a11
            h2 = -a10 - a12 - a01 - a02 - a00 - a22 - a21 - a20 + 9 * a11
            h3 = a10 + a12 + a01 + a02 - 5 * a11
            h4 = a10 + a12 + a01 + a02 + a00 + a22 + a21 + a20 - 9 * a11
            if h1 > 255:
                h1 = 255
            elif h1 < 0:
                h1 = 0
            imgXYH1[i, j] = h1
            if h2 > 255:
                h2 = 255
            elif h2 < 0:
                h2 = 0
            imgXYH2[i, j] = h2
            if h3 > 255:
                h3 = 255
            elif h3 < 0:
                h3 = 0
            imgXYH3[i, j] = h3
            if h4 > 255:
                h4 = 255
            elif h4 < 0:
                h4 = 0
            imgXYH4[i, j] = h4
    files = [img, imgXYH1, imgXYH2,imgXYH3, imgXYH4]
    titles = ['Original', 'imgXYH1', 'imgXYH2','imgXYH3','imgXYH4']
    displays(2,3,titles, files)

def Laplace(H,W,img):
    #H1=np.mat([[0,1,0],        H2 =[[1,1,1]
    #           [1,-4,1],            [1,-8,1]
    #           [0,1,0]])            [1,1,1]]
    imgXYH1 = np.zeros((H,W),np.uint8)
    imgXYH2 = np.zeros((H,W),np.uint8)
    for i in range(1,H-2) :
        for j in range (1,W-2):
            a00 = np.int16(img[i - 1, j - 1])
            a01 = np.int16(img[i - 1, j])
            a02 = np.int16(img[i - 1, j + 1])
            a10 = np.int16(img[i, j - 1])
            a11 = np.int16(img[i, j])
            a12 = np.int16(img[i, j + 1])
            a20 = np.int16(img[i + 1, j - 1])
            a21 = np.int16(img[i + 1, j])
            a22 = np.int16(img[i + 1, j + 1])
            h1 = a10+a12+a01+a02-4*a11
            h2 = a10+a12+a01+a02+a00+a22+a21+a20 -8*a11
            if h1>255:
                h1=255
            elif h1<0:
                h1=0
            imgXYH1[i,j] = h1
            if h2 > 255:
                h2 = 255
            elif h2 < 0:
                h2 = 0
            imgXYH2[i,j] = h2
    files = [img, imgXYH1,imgXYH2]
    titles = ['Original','imgXYH1','imgXYH2']
    displays(1,3,titles, files)

if __name__ == '__main__':
    root = Tk()

    default_dir = r"C:\Users\19845\PycharmProjects\Project"
    File = filedialog.askopenfilename(parent=root, initialdir=(
        os.path.expanduser(default_dir)), title='选择文件')
    print(File)

    img = cv.imread(File)
    img1 = grey(img)
    H = img1.shape[0]
    W = img1.shape[1]
    print(img1.shape)
    cv.imshow("img1",img1)

    # oriImg = cv.imread('cell.jpg', 0)
    # refImg = cv.imread('C:/Users/19845/Desktop/01original.jpg', 0)
    # outImg = histMatching(oriImg, refImg)
    #
    # files = [refImg,oriImg,outImg]
    # titles = ['refImg','oriImg','outImg']
    # displays(1,3,titles,files)

    # huidushangyi(img1)
    # huiduxiayi(img1)
    # huiduduibidu(img1)
    # gama(img1)
    # erzhihua(img1)

    img4 = huidufanzhuan(img1)

    # imgh1 = Hist(img1)
    imgh2 = Hist(img4)
    # cv.imshow("im/gh1",imgh1)
    cv.imshow("imgh2",imgh2)

    # imgN1 = cross3x3(imgh1)
    imgN2 = cross3x3(imgh2)
    # cv.imshow("imgN1",imgN1)
    cv.imshow("imgN2",imgN2)
    #
    # Grad(H, W, imgN1)
    # Robert(H, W, imgN1)
    # Prewitt(H, W, imgN1)
    # Sobel(H, W, imgN1)
    # Synthetic(H,W,imgN1)
    # Laplace(H,W,imgN1)

    Grad(H, W, imgN2)
    Robert(H, W, imgN2)
    Prewitt(H, W, imgN2)
    Sobel(H, W, imgN2)
    Synthetic(H,W,imgN2)
    Laplace(H,W,imgN2)


    cv.waitKey(0)
    cv.destroyAllWindows()
    root.mainloop()

