import cv2
import numpy as np

img = cv2.imread('luotuo.jpg', 0)
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
for i in range(2000):
    _x = np.random.randint(0, th2.shape[0])
    _y = np.random.randint(0, th2.shape[1])
    th2[_x][_y] = 255
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.morphologyEx (th2, cv2.MORPH_OPEN, kernel)
cv2.imshow('th2', th2)
cv2.imshow('morph_open', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()