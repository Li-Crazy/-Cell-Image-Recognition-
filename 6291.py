import cv2
#
# Find Contour
def draw1(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)

    # 需要搞一个list给cv2.drawContours()才行！！！！！
    count = 0
    ares_avrg = 0
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < 200):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        count += 1
        ares_avrg += area
        c_max.append(cnt)

        print("{}-blob:{}".format(count, area), end="  ")  # 打印出每个米粒的面积

        rect = cv2.boundingRect(cnt)  # 提取矩形坐标

        print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标

        cv2.rectangle(img, rect, (0, 0, 0xff), 1)  # 绘制矩形

        y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外

        cv2.putText(img, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX,
                    0.4, (0, 255, 0), 1)  # 在米粒左上角写上编号

    print("米粒平均面积:{}".format(round(ares_avrg / area, 2)))  # 打印出每个米粒的面积

    cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)

    cv2.imshow('mask', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    imgfile = "imgR.jpg"
    img = cv2.imread(imgfile)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("gray1", thresh)

    draw(thresh)




