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
def gray(img):
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

#灰度对比度
def contrast(img,value1,value2):
    image = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if img[i,j] > value1:
                gray = img[i,j]*1.2
                if gray > 255:
                    gray = 255
            elif img[i,j] < value2:
                gray = img[i,j]*0.5
            else:
                gray = img[i,j]
            image[i,j]=gray
    flies = [img,image]
    dis("Contrast",2,files=flies)
    return image

#灰度反转
def reverse(img):
    image = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            gray = 255 - img[i,j]
            image[i,j]=gray
    flies = [img,image]
    dis("Reverse",2,files=flies)
    return image

#二值化
def binarization(img,value):
    image = np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            if img[i,j]>value:
                gray = 255
            else:
                gray = 0
            image[i,j]=gray
    flies = [img,image]
    dis("Binarization",2,files=flies)
    return image

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

#5x5
def median5x5(img):
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
def draw1(thresh,value):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)

    count = 0
    ares_avrg = 0
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        print(area)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < value):
            c_min = []
            c_min.append(cnt)
            cv.drawContours(thresh, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        count += 1
        ares_avrg += area
        c_max.append(cnt)

        rect = cv.boundingRect(cnt)  # 提取矩形坐标
        cv.rectangle(img, rect, (0, 0, 0xff), 1)  # 绘制矩形
        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
        cv.putText(img, str(count), (rect[0], y), cv.FONT_HERSHEY_COMPLEX,
                    0.4, (255, 0, 0), 1)  # 在细胞左上角写上编号
        print("{}-cell:{}".format(count, area), end="  ")  # 打印出每个细胞的面积
        print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标

    cv.drawContours(img, c_max, -1, (0, 0, 0), thickness=1)

    cv.imshow('mask', img)
    cv.waitKey(0)

#正常图片操作
def main1(img):
    imgh1 = Hist(img)#灰度均衡化
    cv.imshow("Hist",imgh1)

    imgN1 = median5x5(imgh1)#中值滤波处理噪声
    cv.imshow("Median5x5", imgN1)

    imgR = Robert(H, W, imgN1)#Robert锐化算子进行边缘检测
    cv.imshow("Robert", imgR)

    imgErZhihua = binarization(imgR,39)#图像二值化
    cv.imshow("Binarization",imgErZhihua)

    fill_img = fillHole(imgErZhihua)#填充二值化图像
    cv.imshow("Fill", fill_img)

    draw1(fill_img,150)#勾画轮廓

    cv.waitKey(0)
    cv.destroyAllWindows()

#对图片进行灰度反转
def main2(img):
    img4 = reverse(img)
    cv.imshow("Reverse",img4)

    imgh1 = Hist(img4)
    cv.imshow("Hist",imgh1)

    imgN1 = median5x5(imgh1)
    cv.imshow("Median5x5", imgN1)

    imgR = Robert(H, W, imgN1)
    cv.imshow("Robert", imgR)

    imgErZhihua = binarization(imgR,39)
    cv.imshow("Binarization",imgErZhihua)

    fill_img = fillHole(imgErZhihua)
    cv.imshow("Fill", fill_img)

    draw1(fill_img,170)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    root = Tk()#创建图形用户界面，用来打开图片
    default_dir = r"C:\Users\19845\PycharmProjects\Project\image"
    #默认打开的文件夹，可修改为目标图片所在文件夹
    File = filedialog.askopenfilename(parent=root, initialdir=(
        os.path.expanduser(default_dir)), title='选择文件')
    print(File)

    img = cv.imread(File)#获取图形用户界面得到的图片路径

    # img = cv.imread("cell3.jpg")#如果无法通过图像用户界面打开，可注释掉318-325行，使用该行获取图片

    cv.imshow("Original",img)

    img1 = gray(img)#图片灰度化
    H = img1.shape[0]
    W = img1.shape[1]
    print(img1.shape)
    cv.imshow("Grayscale",img1)

    img2 = contrast(img1,127,70)#灰度图调整灰度对比度
    cv.imshow("Contrast",img2)

    main1(img2)#正常处理图片

    # main2(img2)#将图片进行灰度反转处理

    cv.waitKey(0)
    cv.destroyAllWindows()