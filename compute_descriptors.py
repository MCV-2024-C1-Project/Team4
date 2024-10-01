import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

qsd1 = "/home/yeray142/Desktop/Team4/data/qsd1_w1"

with open(qsd1 + "/gt_corresps.pkl", 'rb') as f:
    y = pickle.load(f)  # Ground truth
    print(y)

def compute_descriptors(imgs_path):
    for (dirname, dirs, files) in os.walk(imgs_path):
        for filename in files:
            if filename.endswith('.jpg'):
                # Read image
                img = cv.imread(os.path.join(imgs_path, filename))

                # Change color space
                img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
                img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

                # Compute histograms
                hist_lab = cv.calcHist([img_lab], [0], None, [100], [0, 100])
                hist_hsv = cv.calcHist([img_hsv], [0], None, [256], [0, 256])

                #plt.plot(hist_lab)
                print(hist_lab)
                plt.bar(range(len(hist_lab)), hist_lab[0])
                plt.show()


compute_descriptors(qsd1)



