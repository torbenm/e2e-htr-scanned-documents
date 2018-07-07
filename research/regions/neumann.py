"""
Neumann L., Matas J.: 
Real-Time Scene Text Localization and Recognition, 
CVPR 2012
"""
# %%##########################################
#   Initialize Vars                         #
#############################################
import os
print("Working in", os.getcwd())

import cv2
import numpy as np
from matplotlib import pyplot as plt

BASE_PATH = "./research/regions/"
IMAGE_PATH = BASE_PATH + "scene02.jpg"

img = cv2.imread(IMAGE_PATH)
vis = img.copy()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

channels = cv2.text.computeNMChannels(img)
cn = len(channels)-1
for c in range(0, cn):
    channels.append((255-channels[c]))
print("Found "+str(len(channels))+" channels ...")

for channel in channels:

    #erc1 = cv2.text.loadClassifierNM1(BASE_PATH + 'trained_classifierNM1.xml')
    er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

    #erc2 = cv2.text.loadClassifierNM2(BASE_PATH + '/trained_classifierNM2.xml')
    er2 = cv2.text.createERFilterNM2(erc2, 0.5)

    print(er2)

    regions = cv2.text.detectRegions(channel, er1, er2)

    rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])

    # Visualization
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        cv2.rectangle(vis, (rect[0], rect[1]),
                      (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 0), 2)
        cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] +
                                                rect[2], rect[1]+rect[3]), (255, 255, 255), 1)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
