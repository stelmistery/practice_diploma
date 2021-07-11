import cv2
import numpy as np
img = cv2.imread('luotuo.jpg', 0)
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
npKernel = np.uint8(np.zeros((5, 5)))
for i in range(5):
    npKernel[2, i] = 1
    npKernel[i, 2] = 1
print(npKernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
npKernel_eroded = cv2.erode(th2, npKernel)
kernel_eroded = cv2.erode(th2, kernel)
cv2.imshow('img', th2)
cv2.imshow('npKernel Eroded Image', npKernel_eroded)
cv2.imshow('kernel Eroded Image', kernel_eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
