import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from enum import Enum
from utils import plot_hist_from_img, plot_hist_from_list
from metrics import compare_histograms

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory
qsd1 = os.path.join(base_path, "data", "qsd1_w1")
bbdd = os.path.join(base_path, "data", "BBDD")

with open(qsd1 + "/gt_corresps.pkl", 'rb') as f:
	y = pickle.load(f)  # Ground truth


def compute_histogram(img, channels, bins, ranges):
	# Compute histogram and normalize
	hist = cv.calcHist([img], channels, None, bins, ranges)
	cv.normalize(hist, hist)
	return hist.flatten()


def compute_descriptors(imgs_path):
	# Array to store histograms
	lab_histograms = []
	hsv_histograms = []

	# Get the list of files and sort them by the number in the filename
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

	for filename in files:
		if filename.endswith('.jpg'):
			# Read image
			img = cv.imread(os.path.join(imgs_path, filename))

			# Change color space
			img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
			img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

			# Print histograms (just if needed)
			#plot_hist_from_img(img_lab, ['L channel', 'a channel', 'b channel'], filename)
			#plot_hist_from_img(img_hsv, ["Hue channel", "Saturation channel", "Value channel"], filename, color="green")

			# Compute histograms for each channel and concatenate them
			lab_hist = np.concatenate(
				[compute_histogram(img_lab, [0], [256], [0, 256]),
				 compute_histogram(img_lab, [1], [256], [0, 256]),
				 compute_histogram(img_lab, [2], [256], [0, 256])]
			)

			hsv_hist = np.concatenate(
				[compute_histogram(img_hsv, [0], [180], [0, 180]),
				 compute_histogram(img_hsv, [1], [256], [0, 256]),
				 compute_histogram(img_hsv, [2], [256], [0, 256])]
			)


			# Extract the index from the filename and store histogram at the correct position
			index = int(filename.split('_')[-1].split('.')[0])

			if len(lab_histograms) <= index:
				lab_histograms.extend([None] * (index + 1 - len(lab_histograms)))
			lab_histograms[index] = lab_hist

			if len(hsv_histograms) <= index:
				hsv_histograms.extend([None] * (index + 1 - len(hsv_histograms)))
			hsv_histograms[index] = hsv_hist

			# Show histograms (just if needed)
			# plot_hist_from_list(lab_histograms, index, color_space='Lab')

	# print(compare_histograms(lab_histograms[29], lab_histograms[24]))

	# Save histograms to a pickle file
	with open(os.path.join(imgs_path, 'lab_histograms.pkl'), 'wb') as f:
		pickle.dump(lab_histograms, f)
	with open(os.path.join(imgs_path, 'hsv_histograms.pkl'), 'wb') as f:
		pickle.dump(hsv_histograms, f)


compute_descriptors(bbdd)
#compute_descriptors(qsd1)