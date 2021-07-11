import numpy as np
import cv2
from matplotlib import pyplot as pltfrom
import matplotlib.pyplot as plt
image = cv2.imread('../color/tool.jpg') # reads the image
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR) # convert to HSV
figure_size = 9 # the dimension of the x and y axis of the kernal.
new_image = cv2.blur(image,(figure_size, figure_size))
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB))
plt.title('Mean filter')
plt.xticks([])
plt.yticks([])
plt.show()

image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
figure_size = 9
new_image = cv2.blur(image2,(figure_size, figure_size))
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(image2, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(new_image, cmap='gray')
plt.title('Mean filter')
plt.xticks([])
plt.yticks([])
plt.show()