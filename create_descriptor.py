#Usage: python create_descriptor.py  <list of training images>

import sys
import os
import cv2

from utility_functions import *

trainLst = sys.argv[1]

trainFile = open (trainLst)

for i in trainFile:
    i = i[:-1] #remove new line character
    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(len(keypoints), end=" ")
    print("keypoints detected")
    descriptorFileName = i[:-3] + 'decp'
    saveDescriptorListFile(descriptors, descriptorFileName)

trainFile.close()