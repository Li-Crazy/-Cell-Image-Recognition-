from tkinter import *
from tkinter import filedialog
import cv2 as cv
import os
from PIL import Image, ImageTk


def printcoords():
    default_dir = r"C:\Users\19845\PycharmProjects\Project"
    File = filedialog.askopenfilename(parent=root, initialdir=(
    os.path.expanduser(default_dir)),title='选择文件')
    print(File)
    img = cv.imread(File)
    cv.imshow("File", img)
    cv.waitKey(0)

if __name__ == "__main__":
    root = Tk()
    Button(root,text='选择图片',command=printcoords).pack()
    root.mainloop()
