import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from tqdm import tqdm

def binaryImg():
    image = cv2.imread("canny/imp1.jpg", 0)
    h = image.shape[0]
    w = image.shape[1]

    a = 150
    new_image = np.zeros((h,w),np.uint8)
    for i in tqdm(range(h)):
        for j in range(w):
            if(image[i,j]> a ):
                new_image[i,j] = 255
            else:
                new_image[i,j] = 0

    print(new_image)
    cv2.imshow("new",new_image)
    cv2.waitKey()

def adaptiveBinaryImg():
    original_image = cv2.imread("canny/imp1.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)  # Бинарный порог
    # Порог берется из среднего значения соседних областей
    # thresh_mean = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 0)
    # Порог берется из взвешенной суммы смежных областей, а вес является гауссовым окном.
    thresh_gaussian = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 0)

    # images = [original_image, binary, thresh_mean, thresh_gaussian]
    # titles = ['Original Image', 'BINARY', 'MEAN', 'GAUSSIAN']

    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])  # Отмена шкалы оси
    # plt.show()
    plt.figure(dpi=700)
    plt.plot(1, 1, 1)
    plt.imshow(thresh_gaussian, "gray")
    plt.show()

if __name__ == '__main__':
    # binaryImg()
    adaptiveBinaryImg()