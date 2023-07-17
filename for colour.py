import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
lower_red = np.array([0, 127, 128])  # 红色阈值下界
higher_red = np.array([10, 255, 255])  # 红色阈值上界
lower_green = np.array([35, 110, 106])  # 绿色阈值下界
higher_green = np.array([77, 255, 255])  # 绿色阈值上界
cap = cv2.VideoCapture(0)  # 打开电脑内置摄像头
if (cap.isOpened()):
    while (True):
        ret, frame = cap.read()  # 按帧读取，这是读取一帧
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(img_hsv, lower_red, higher_red)  # 可以认为是过滤出红色部分，获得红色的掩膜
        mask_green = cv2.inRange(img_hsv, lower_green, higher_green)  # 获得绿色部分掩膜
        mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
        mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波
        mask = cv2.bitwise_or(mask_green, mask_red)  # 三部分掩膜进行按位或运算
        cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
        cnts3, hierarchy3 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
            cv2.putText(frame, 'red', (x, y - 5), font, 0.7, (0, 0, 255), 2)

        for cnt in cnts3:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
            cv2.putText(frame, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

cv2.waitKey(0)
cv2.destroyAllWindows()