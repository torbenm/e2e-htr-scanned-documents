from matplotlib import pyplot
import cv2


def plt_gray(img):
    pyplot.imshow(img, cmap='gray')


def plt_bgr(img):
    pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
