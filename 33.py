'''
-*- coding: utf-8 -*-
@Author  : LiZhichao
@Time    : 2019/6/26 15:40
@Software: PyCharm
@File    : 33.py
'''
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import cv2 as cv
import os

root=tk.Tk()
root.title('我的第一个python窗体')     #title
root.geometry('320x240')              #这里的乘号是英文字母x,不是*

def openImage():
    root = tk.Tk()  # 创建一个Tkinter.Tk()实例
    root.withdraw()  # 将Tkinter.Tk()实例隐藏
    default_dir = r"C:\Users\19845\PycharmProjects\Project"
    file_path = tk.filedialog.askopenfilename(title=u'选择文件', initialdir=(
    os.path.expanduser(default_dir)))
    print(file_path)
    img = cv.imread(file_path)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#用command指定回调函数
Button(root,text='选择图片',command=openImage).pack()
root.mainloop()        #运行窗口，创建GUI根窗体

