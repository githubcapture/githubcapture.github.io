
import cv2
import numpy as np
import torch

torch.cuda.is_available()

tag_size = 0.05
tag_size_half = 0.025
fx = 1052.66732619183                                    #自测内参矩阵
fy = 1050.74512910951
cx = 972.934540675688
cy = 575.994600292495
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)
objPoints = np.array([[2, 1, 4],
                      [4, 6, 7],
                      [1, 2, 5],
                      [8, 1, 3]], dtype=np.float64)       #非平面最好六个点 世界坐标确定点

imgPoints = np.array([[608, 167], [514, 167], [518, 69], [611, 71]], dtype=np.float64)
                                                          #对应点的相机像素坐标
cameraMatrix = K
distCoeffs = np.array([[-0.0510351047175117],[0.155379636349165],[0],[0],[0]], dtype=np.float64)
retval,rvec,tvec  = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
rot_mat, _ = cv2.Rodrigues(rvec)                          # 把旋转向量转换为旋转矩阵
a,b,c = np.vsplit(rot_mat.T, 3)
#print(rot_mat,tvec)
q = np.array(a)
w = np.array(b)
e = np.array(c)
r = np.array([0.0,0.0,0.0,1.0])

x = np.vstack((q,w))
y = np.vstack((x,e))
z = np.vstack((y,tvec.T))

A,B,C = np.vsplit(z.T, 3)
X = np.vstack((A,B))
Y = np.vstack((X,C))
K = np.vstack((Y,r))

r_2 = np.array([1])

sj = [[5],[6],[7],[1]]

'''
sj1,sj2,sj3 = np.vsplit(sj.T,3)
x_1 = np.array((sj1,sj2))
y_1 = np.array((x_1,sj2))
z_1 = np.array((y_1,sj3))
r_1 = np.array((z_1,r_2))
print(z_1.T)'''

l = np.array(K)
print(tvec)
print(np.dot(l, sj))