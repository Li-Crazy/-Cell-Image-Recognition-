import cv2
import numpy as np


def FillHole(imgPath, SavePath):
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

def FillHole_RGB(imgPath, SavePath, SizeThreshold):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)

    # 将im_in_rgb的RGB颜色转换为 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (
    im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0

        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (m, n)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        # 得到im_floodfill的逆im_floodfill_inv，im_floodfill_inv包含所有孔洞
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 之所以复制一份im_floodfill_inv是因为函数findContours会改变im_floodfill_inv_copy
        im_floodfill_inv_copy = im_floodfill_inv.copy()
        # 函数findContours获取轮廓
        contours, hierarchy = cv2.findContours(im_floodfill_inv_copy,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= SizeThreshold):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3),
                           np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    cv2.imwrite(SavePath, im_fillhole)

if __name__ == '__main__':
    imgPath = "cell3.jpg"
    SavePath = "cell4.jpg"
    FillHole_RGB(imgPath,SavePath,50)