import pickle
import os
import cv2 as cv

from compute_similarities import compute_similarities
from compute_descriptors import compute_descriptors
from average_precision import mapk

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory
qsd1 = os.path.join(base_path, "data", "qsd1_w1")
bbdd = os.path.join(base_path, "data", "BBDD")

with open(qsd1 + "/gt_corresps.pkl", 'rb') as f:
	y = pickle.load(f)  # Ground truth


compute_descriptors(qsd1)

with open(os.path.join(qsd1, 'lab_histograms.pkl'), 'rb') as f:
    qsd1_lab_histograms = pickle.load(f)
with open(os.path.join(qsd1, 'hsv_histograms.pkl'), 'rb') as f:
    qsd1_hsv_histograms = pickle.load(f)

with open(os.path.join(bbdd, 'hsv_histograms.pkl'), 'rb') as f:
    bbdd_hsv_histograms = pickle.load(f)
with open(os.path.join(bbdd, 'lab_histograms.pkl'), 'rb') as f:
    bbdd_lab_histograms = pickle.load(f)

# Method 1 - Lab - Hellinger
k_value = 5

res_m1_k1 = []
for query_lab_img in qsd1_lab_histograms:
    res_m1_k1.append(compute_similarities(query_hist=query_lab_img, bbdd_histograms=bbdd_lab_histograms, method=cv.HISTCMP_HELLINGER, k=k_value)[1])
print(mapk(y, res_m1_k1, k_value))
#print(res_m1_k1)

# Method 2 - HSV - Hellinger
res_m2_k1 = []
for query_hsv_img in qsd1_hsv_histograms:
    res_m2_k1.append(compute_similarities(query_hist=query_hsv_img, bbdd_histograms=bbdd_hsv_histograms, method=cv.HISTCMP_HELLINGER, k=k_value)[1])
print(mapk(y, res_m2_k1, k_value))




