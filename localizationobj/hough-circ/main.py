#!python
# -*- coding: utf-8 -*-
__version__ = "$Revision: 1.8 $"
# $Source: /home/mechanoid/projects/py/cv/_howse2/contour3-hough-circle/RCS/main.py,v $
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
#       OS : GNU/Linux 4.10.3-1-ARCH 
# COMPILER : Python 3.6.0
#
#   AUTHOR : Evgeny S. Borisov
# 
#    http://www.mechanoid.kiev.ua
#  e-mail : nn@mechanoid.kiev.ua
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import cv2
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def main():
    img = cv2.imread("data/pla.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)
    cv2.imwrite("result/01-blur.jpg",gray)

    edges = cv2.Canny(gray,50,200)
    cv2.imwrite("result/02-edge.jpg", edges)

    img = cv2.imread('result/02-edge.jpg', 0)
    # img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == "__main__":
    print("OpenCV ",cv2.__version__)
    sys.exit(main())


