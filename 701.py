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

def displays(Name,r,c,titles,files):
    plt.figure(Name)
    for i in range(len(files)):
        plt.subplot(r,c,i+1)
        plt.imshow(files[i], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(titles[i])
    plt.show()

#灰度化
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
    return img2

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
    return img2

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
            if img1[i,j]>38:
                gray = 255
            else:
                gray = 0
            img6[i,j]=gray
    flies = [img1,img6]
    dis("erzhihua",2,files=flies)
    return img6

#直方图均衡化
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
    displays("Hist",1,2,titles,files)
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
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0:
                newimg[i, j] = img[i, j]
            elif j == 0:
                newimg[i, j] = img[i, j]
            elif i == H - 1:
                newimg[i, j] = img[i, j]
            elif j == W - 1:
                newimg[i, j] = img[i, j]
            else:
                newimg[i, j] = curimg[i, j]
    files = [img,curimg,newimg]
    dis('3x3', 3, files)
    return newimg

#5x5
def cross5x5(img):
    curimg = np.zeros((H,W),np.uint8)
    newimg = np.zeros((H,W),np.uint8)

    for i in range(1,H-2):
        for j in range(1,W-2):
            t00 = img[i-2,j-2]
            t01 = img[i-2,j-1]
            t02 = img[i-2,j]
            t03 = img[i-2,j+1]
            t04 = img[i-2,j+2]
            t10 = img[i-1, j - 2]
            t11 = img[i-1, j - 1]
            t12 = img[i-1, j]
            t13 = img[i-1, j + 1]
            t14 = img[i-1, j + 2]
            t20 = img[i , j - 2]
            t21 = img[i , j - 1]
            t22 = img[i , j]
            t23 = img[i , j + 1]
            t24 = img[i , j + 2]
            t30 = img[i+1, j - 2]
            t31 = img[i+1, j - 1]
            t32 = img[i+1, j]
            t33 = img[i+1, j + 1]
            t34 = img[i+1, j + 2]
            t40 = img[i+2, j - 2]
            t41 = img[i+2, j - 1]
            t42 = img[i+2, j]
            t43 = img[i+2, j + 1]
            t44 = img[i+2, j + 2]
            templ =[t00,t01,t02,t03,t04,t10,t11,t12,t13,t14,t20,t21,t22,t23,t24,t30,t31,t32,t33,t34,t40,t41,t42,t43,t44]
            templ.sort()
            curimg[i,j]=templ[12]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0:
                newimg[i, j] = img[i, j]
            elif j == 0:
                newimg[i, j] = img[i, j]
            elif i == H-1 or i == H-2:
                newimg[i, j] = img[i, j]
            elif j == W-1 or j==W-2:
                newimg[i, j] = img[i, j]
            else:
                newimg[i, j] = curimg[i, j]
    files = [img, curimg, newimg]
    dis('5x5', 3, files)
    return newimg

#Robert算子
def Robert(H,W,img):
    imgR = np.zeros((H,W),np.uint8)
    for i in range(0, H - 1):
        for j in range(0, W - 1):
            a00 = np.int16(img[i,j])
            a01 = np.int16(img[i,j+1])
            a10 = np.int16(img[i+1,j])
            a11 = np.int16(img[i+1,j+1])
            # u = np.sqrt((a00-a11)**2 + (a10-a01)**2)
            u = np.abs((a00-a11))+np.abs(a10-a01)
            if u>= 255:
                u = 255
            elif u<0:
                u = 0
            imgR[i,j] =np.uint8(u)
    files = [img,imgR]
    dis('Robert', 2, files)
    return imgR

#梯度
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
    return imgG

#Prewitt算子
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
    displays("Prewitt",1,3,titles,files)
    return imgpXY, imgpS

# Sobel算子
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
    displays("Sobel",1,3,titles,files)
    return imgXandY,imgabS

#合成
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
    displays("Synthetic",2,3,titles, files)
    return imgXYH1, imgXYH2,imgXYH3, imgXYH4

#拉普拉斯
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
    displays("Laplace",1,3,titles, files)
    return imgXYH1,imgXYH2

def Lookformin(subimg):
    (row,col)=subimg.shape
    min = 0
    for i in range(row):
        for j in range(col):
            if subimg[i,j]>0:
                min =subimg[i,j]
                break
    for i in range(row):
        for j in range(col):
            if subimg[i, j] < min and subimg[i,j]>0:
                min = subimg[i, j]
    return min

#腐蚀
def getErode(img, kernel):
    (H,W) =img.shape
    erodeimg2 = np.zeros((H,W),np.uint8)
    print ("img.shape is (",H,W,")")
    (row,col)=kernel.shape
    print("kernel.shape(",row,col,")=\n", kernel)
    # 等于结构元素的子图像
    subimg = np.zeros((row, col), np.uint8)
    # 结构元素卷积运算原始图像
    for h in range(row//2, H-row//2):
        for w in range(col//2, W-col//2):
            #结构元素与对应点
            for i in range(-(row//2), row//2+1):
                for j in range(-(col//2), col//2+1):
                    subimg[row//2+i,col//2+j] = img[h+i,w+j]*kernel[row//2+i,col//2+j]

            smin = Lookformin(subimg)
            erodeimg2[h,w]=smin
    return erodeimg2

#膨胀
def getDilate(img, kernel):
    (H,W) =img.shape
    dilate2 = np.zeros((H, W), np.uint8)
    print("img.shape(H,W)=", H, W)
    (row,col)=kernel.shape
    # 等于结构元素的子图像
    subimg = np.zeros((row, col), np.uint8)
    # 结构元素卷积运算原始图像
    print("kernel.shape(row,col)", row, col)
    for h in range(row//2, H-row//2):
        for w in range(col//2, W-col//2):
            #结构元素与对应点
            for i in range(-(row//2), row//2+1):
                for j in range(-(col//2), col//2+1):
                    subimg[row//2+i,col//2+j] = img[h+i,w+j]*kernel[row//2+i,col//2+j]
            smax = subimg.max()
            dilate2[h, w] = smax
    return dilate2

#填充
def fillHole(im_in):
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    return im_out

#绘制轮廓
def draw(thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)

    # 需要搞一个list给cv2.drawContours()才行！！！！！
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        print(area)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < 200):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv.drawContours(thresh, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        #
        c_max.append(cnt)

    # print(c_max)

    cv.drawContours(img, c_max, -1, (0, 0, 0), thickness=1)

    # cv2.imwrite("mask.png", img)
    cv.imshow('mask', img)
    cv.waitKey(0)

def draw1(thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)

    # 需要搞一个list给cv2.drawContours()才行！！！！！
    count = 0
    ares_avrg = 0
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        print(area)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < 150):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv.drawContours(thresh, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        count += 1
        ares_avrg += area
        c_max.append(cnt)

        # print("{}-blob:{}".format(count, area), end="  ")  # 打印出每个细胞的面积

        rect = cv.boundingRect(cnt)  # 提取矩形坐标

        # print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标

        cv.rectangle(img, rect, (0, 0, 0xff), 1)  # 绘制矩形

        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外

        cv.putText(img, str(count), (rect[0], y), cv.FONT_HERSHEY_COMPLEX,
                    0.4, (255, 0, 0), 1)  # 在细胞左上角写上编号

    cv.drawContours(img, c_max, -1, (0, 0, 0), thickness=1)

    cv.imshow('mask', img)
    cv.waitKey(0)

def main1(img):
    # oriImg = cv.imread('cell.jpg', 0)
    # refImg = cv.imread('C:/Users/19845/Desktop/01original.jpg', 0)
    # outImg = histMatching(oriImg, refImg)
    #
    # files = [refImg,oriImg,outImg]
    # titles = ['refImg','oriImg','outImg']
    # displays(1,3,titles,files)

    imgh1 = Hist(img)
    cv.imshow("imgh1",imgh1)
    imgN1 = cross5x5(imgh1)
    cv.imshow("5x5", imgN1)

    imgR = Robert(H, W, imgN1)
    cv.imshow("imgR", imgR)

    kernelE = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                       dtype=np.uint8)
    kernelD = np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]], dtype=np.uint8)

    dilate_img = getDilate(imgR, kernelD)
    cv.imshow("Dilate", dilate_img)
    erode_img = getErode(dilate_img, kernelE)
    cv.imshow("Erode", erode_img)

    imgErZhihua = erzhihua(imgR)

    fill_img = fillHole(imgErZhihua)
    cv.imshow("Fill", fill_img)

    draw1(fill_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def main2(img):
    # oriImg = cv.imread('cell.jpg', 0)
    # refImg = cv.imread('C:/Users/19845/Desktop/01original.jpg', 0)
    # outImg = histMatching(oriImg, refImg)
    #
    # files = [refImg,oriImg,outImg]
    # titles = ['refImg','oriImg','outImg']
    # displays(1,3,titles,files)

    img4 = huidufanzhuan(img)
    imgh1 = Hist(img4)
    # cv.imshow("imgh1",imgh1)
    imgN1 = cross5x5(imgh1)
    cv.imshow("5x5", imgN1)

    imgR = Robert(H, W, imgN1)
    cv.imshow("imgR", imgR)

    kernelE = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                       dtype=np.uint8)
    kernelD = np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]], dtype=np.uint8)

    dilate_img = getDilate(imgR, kernelD)
    cv.imshow("Dilate", dilate_img)
    erode_img = getErode(dilate_img, kernelE)
    cv.imshow("Erode", erode_img)

    imgErZhihua = erzhihua(imgR)

    fill_img = fillHole(imgErZhihua)
    cv.imshow("Fill", fill_img)

    draw1(fill_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # root = Tk()
    # default_dir = r"C:\Users\19845\PycharmProjects\Project"
    # File = filedialog.askopenfilename(parent=root, initialdir=(
    #     os.path.expanduser(default_dir)), title='选择文件')
    # print(File)

    # img = cv.imread(File)
    img = cv.imread("cell3.jpg")
    img1 = grey(img)
    H = img1.shape[0]
    W = img1.shape[1]
    print(img1.shape)
    cv.imshow("img1",img1)

    main1(img1)
    # main2(img1)







