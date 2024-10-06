import cv2 as cv
import numpy as np
import pickle
import os

from utils import plot_hist_from_img, plot_hist_from_list
from metrics import compare_histograms


def compute_histogram(img, channels, bins, ranges, normalized=True) -> np.ndarray:
	"""
	Computes the histogram of an image
	:param img: image
	:param channels: channels of the color space for which the histogram has to be computed
	:param bins: number of bins (number of levels of the histogram)
	:param ranges: range of values
	:param normalized: if True, normalize the histogram. Default is False
	:return: 1D array containing the computed histogram
	"""

	hist = cv.calcHist([img], channels, None, bins, ranges)
	if normalized:
		cv.normalize(hist, hist)
	return hist.flatten()


def compute_descriptors(imgs_path: str, color_space: str = "HSV") -> None:
	"""
	Compute 1D histograms for each image in the dataset and save them to a '.pkl' file
	:param imgs_path: path to the dataset from which images are extracted
	:param color_space: chosen color space to represent the image (e.g., HSV. Lab, RGB, YCbCr)
	"""

	# Array to save the computed histograms
	histograms = []

	# Get the list of '.jpg' files (images of museum paintings) and sort them by 
	# the numerical ID appended in their filenames
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

	for filename in files:

		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(imgs_path, filename))

		# Change color space (only 2 options are possible)
		if color_space == "Lab": 
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
			bins_channel1 = 256
		elif color_space == "HSV": 
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
			bins_channel1 = 180


		# Print histograms (just if needed)
		# plot_hist_from_img(img, ['channel_1', 'channel_2', 'channel_3'], filename)

		# Compute 1D histograms for each channel and concatenate them

		hist = np.concatenate(
			[compute_histogram(img, [0], [bins_channel1], [0, bins_channel1], normalized=True),
			 compute_histogram(img, [1], [256], [0, 256], normalized=True),
			 compute_histogram(img, [2], [256], [0, 256], normalized=True)]
		)
		# Extract the index (numerical ID) from the filename and store histogram at the correct position
		index = int(filename.split('_')[-1].split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

		# Show histograms (just if needed)
		# plot_hist_from_list(histograms, index, color_space=color_space)

	# print(compare_histograms(histograms[29], histograms[24]))

	# Save histograms to a pickle file
	with open(os.path.join(imgs_path, color_space +'_histograms.pkl'), 'wb') as f:
		pickle.dump(histograms, f)


