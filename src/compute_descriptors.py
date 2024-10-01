import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from utils import print_hist

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory
qsd1 = os.path.join(base_path, "data", "qsd1_w1")
bbdd = os.path.join(base_path, "data", "BBDD")

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

                # Print histograms just if needed
                print_hist(img_lab, ['L channel', 'a channel', 'b channel'], filename)
                print_hist(img_hsv, ["Hue channel", "Saturation channel", "Value channel"], filename, color="green")




compute_descriptors(bbdd)



