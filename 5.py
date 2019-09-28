import cv2

vc = cv2.VideoCapture('C:/Users/19845/Desktop/video.MP4')  # 读入视频文件
c = 1

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 10  # 视频帧计数间隔频率

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if (c % timeF == 0):  # 每隔timeF帧进行存储操作
        cv2.imwrite('image/' + str(c) + '.jpg', frame)  # 存储为图像
    c = c + 1
    cv2.waitKey(1)
vc.release()
