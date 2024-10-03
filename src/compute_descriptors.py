import cv2 as cv
import numpy as np
import pickle
import os

from utils import plot_hist_from_img, plot_hist_from_list
from metrics import compare_histograms


def compute_histogram(img, channels, bins, ranges, normalized=False):
	"""
	Computes the histogram of an image
	:param img: image
	:param channels: channels to compute the histogram
	:param bins: number of bins
	:param ranges: range of values
	:param normalized: if True, normalize the histogram. Default is False
	"""
	hist = cv.calcHist([img], channels, None, bins, ranges)
	if normalized:
		cv.normalize(hist, hist)
	return hist.flatten()

"""
def compute_descriptors(imgs_path):
	""" """
	Compute histograms for each image in the dataset and save them to a pickle file
	:param imgs_path: path to the dataset
	""" """
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
			# plot_hist_from_img(img_lab, ['L channel', 'a channel', 'b channel'], filename)
			# plot_hist_from_img(img_hsv, ["Hue channel", "Saturation channel", "Value channel"], filename, color="green")

			# Compute histograms for each channel and concatenate them
			lab_hist = np.concatenate(
				[compute_histogram(img_lab, [0], [256], [0, 256], normalized=True),
				 compute_histogram(img_lab, [1], [256], [0, 256], normalized=True),
				 compute_histogram(img_lab, [2], [256], [0, 256], normalized=True)]
			)

			hsv_hist = np.concatenate(
				[compute_histogram(img_hsv, [0], [180], [0, 180], normalized=True),
				 compute_histogram(img_hsv, [1], [256], [0, 256], normalized=True),
				 compute_histogram(img_hsv, [2], [256], [0, 256], normalized=True)]
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
"""


def compute_descriptors(imgs_path: str, color_space: str = "HSV"):
	"""
	Compute histograms for each image in the dataset and save them to a pickle file
	:param imgs_path: path to the dataset
	"""
	# Array to store the computed histograms
	histograms = []

	# Get the list of '.jpg' files (images of museum paintings) and sort them by the numerical ID appended in their filenames
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

		# Compute histograms for each channel and concatenate them
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


