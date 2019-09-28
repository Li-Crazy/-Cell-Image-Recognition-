'''
-*- coding: utf-8 -*-
@Author  : LiZhichao
@Time    : 2019/6/26 15:01
@Software: PyCharm
@File    : 11.py
'''

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5.QtCore import QFileInfo


class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.myButton = QtWidgets.QPushButton(self)
        self.myButton.setObjectName("btn")
        self.myButton.setText("按钮")
        self.myButton.clicked.connect(self.msg)


    def msg(self):

        directory1 = QFileDialog.getExistingDirectory(self, "选择文件夹", "/")
        print(directory1)  # 打印文件夹路径

        fileName, filetype = QFileDialog.getOpenFileName(self, "选择文件", "/", "All Files (*);;Text Files (*.txt)")
        print(fileName, filetype)  # 打印文件全部路径（包括文件名和后缀名）和文件类型
        print(fileName)  # 打印文件全部路径（包括文件名和后缀名）
        fileinfo = QFileInfo(fileName)
        print(fileinfo)  # 打印与系统相关的文件信息，包括文件的名字和在文件系统中位置，文件的访问权限，是否是目录或符合链接，等等。
        file_name = fileinfo.fileName()
        print(file_name)  # 打印文件名和后缀名
        file_suffix = fileinfo.suffix()
        print(file_suffix)  # 打印文件后缀名
        file_path = fileinfo.absolutePath()
        print(file_path)  # 打印文件绝对路径（不包括文件名和后缀名）

        files, ok1 = QFileDialog.getOpenFileNames(self, "多文件选择", "/", "所有文件 (*);;文本文件 (*.txt)")
        print(files, ok1)  # 打印所选文件全部路径（包括文件名和后缀名）和文件类型

        fileName2, ok2 = QFileDialog.getSaveFileName(self, "文件保存", "/", "图片文件 (*.png);;(*.jpeg)")
        print(fileName2)  # 打印保存文件的全部路径（包括文件名和后缀名）


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
