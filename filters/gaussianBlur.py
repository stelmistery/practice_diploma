import numpy as np
import cv2
from matplotlib import pyplot as pltfrom
import matplotlib.pyplot as plt

figure_size = 9
image = cv2.imread('../color/tool.jpg')
new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB))
plt.title('Gaussian Filter')
plt.xticks([])
plt.yticks([])
plt.show()

image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
new_image_gauss = cv2.GaussianBlur(image2, (figure_size, figure_size),0)
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(image2, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(new_image_gauss, cmap='gray')
plt.title('Gaussian Filter')
plt.xticks([])
plt.yticks([])
plt.show()
