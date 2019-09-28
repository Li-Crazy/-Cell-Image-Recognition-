import cv2
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    # 如果符合条件，返回True，否则返回False
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 根据坐标画出人物所在的位置
def draw_person(img, person):
  x, y, w, h = person
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 定义HOG特征+SVM分类器
img = cv2.imread("222.jpg")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)

# 判断坐标位置是否有重叠
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        a = is_inside(r, q)
        if ri != qi and a:
            break
    else:
        found_filtered.append(r)
# 勾画筛选后的坐标位置
for person in found_filtered:
    draw_person(img, person)
# 显示图像
cv2.imshow("people detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()