
# Based on
# https://github.com/HsiehYiChia/Scene-text-recognition
#  and
"""
Cho, Hojin, Myungchul Sung, and Bongjin Jun. 
"Canny text detector: Fast and robust scene text localization algorithm." 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
"""
# %%##########################################
#   Initialize Vars                         #
#############################################
import os
print("Working in", os.getcwd())

import cv2
import numpy as np
from matplotlib import pyplot as plt


IMAGE_PATH = "./research/region/firstaid.png"

img = cv2.imread(IMAGE_PATH)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%##########################################
#   Character Candidate Extraction          #
#############################################


def character_candidate_extraction(img):
    pass


character_candidate_extraction(img)
