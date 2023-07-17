
'''
HoughCircles设置
第3参数默认为1
第4参数表示圆心与圆心之间的距离（太大的话，会很多圆被认为是一个圆）
第5参数默认为100
第6参数根据圆大小设置(圆越小设置越小，检测的圆越多，但检测大圆会有噪点)
第7圆最小半径
第8圆最大半径
'''

import cv2
import numpy as np
import pandas as pd

font = cv2.FONT_HERSHEY_SIMPLEX
lower_red = np.array([0, 127, 128])  # 红色阈值下界
higher_red = np.array([10, 255, 255])  # 红色阈值上界
lower_green = np.array([35, 110, 106])  # 绿色阈值下界
higher_green = np.array([77, 255, 255])  # 绿色阈值上界

cap = cv2.VideoCapture(0)  # 0表示第一个摄像头
while (1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    frame1 = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.medianBlur(gray, 9)  # 中值模糊
    th2 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)#二值化需要的参数是src灰度图像 倒数第二个参数越大细节越少
    kernel = np.ones((5, 5), np.uint8)  # 创建全一矩阵，数值类型设置为uint8
    erosion = cv2.erode(th2, kernel, iterations=1)  # 腐蚀处理
    dilation = cv2.dilate(erosion, kernel, iterations=1)  # 膨胀处理
    imgray = cv2.Canny(erosion, 30, 100)  # Canny算子边缘检测

    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 4, 2500, param1=100, param2=90, minRadius=250, maxRadius=258)

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(img_hsv, lower_red, higher_red)  # 可以认为是过滤出红色部分，获得红色的掩膜
    mask_green = cv2.inRange(img_hsv, lower_green, higher_green)  # 获得绿色部分掩膜
    mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
    mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波
    mask = cv2.bitwise_or(mask_green, mask_red)  # 三部分掩膜进行按位或运算
    cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
    cnts3, hierarchy3 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if type(circles) is not type(None):
        circles = np.uint16(np.around(circles))

        P = circles[0]  # 去掉circles数组一层外括号
        for i in P:
            # 画出外圆
            cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 画出圆心
            cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)
        print("圆的个数是：")
        print(len(P))
        for i in P:
            r = int(i[2])
            x = int(i[0])
            y = int(i[1])
            print("圆心坐标为：", (x, y))
            print("圆的半径是：", r)
            print('离兔子的距离', y)

        for cnt in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
            cv2.putText(frame, 'red', (x, y - 5), font, 0.7, (0, 0, 255), 2)
            print('红色圆环')
        for cnt in cnts3:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
            cv2.putText(frame, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
            print('绿色圆环')

        cv2.imshow('frame', frame)
    cv2.imshow('detected circles', gray)
    if type(circles) is type(None):
        print('picture without circles')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
