import cv2
import numpy as np
#import image
# https://stackoverflow.com/questions/46282691/opencv-cropping-handwritten-lines-line-segmentation


def show(cap):
    global img
    cv2.imshow(cap, img)
    cv2.waitKey(0)


img_org = cv2.imread('input.jpg')
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)


mean = np.mean(img, axis=(0, 1))
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 11, mean/4)
show('binarized')

# (5,1) for input2
img = cv2.erode(img, np.ones((5, 1)), iterations=1)
show('eroded')

# (5, 50) for input2
img = cv2.dilate(img, np.ones((5, 50)), iterations=1)
show('dilated')

# find contours
im2, ctrs, hier = cv2.findContours(
    img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    if w / h < 1:
        continue

    # Getting ROI
    roi = img_org[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i), roi)
    cv2.rectangle(img_org, (x, y), (x + w, y + h), (90, 0, 255), 2)
    # cv2.waitKey(0)

cv2.imshow('marked areas', img_org)
cv2.waitKey(0)
