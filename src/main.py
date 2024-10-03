import pickle
import os
import cv2 as cv

from compute_similarities import compute_similarities
from compute_descriptors import compute_descriptors
from average_precision import mapk
from utils import plot_hist_task1

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qsd1 = os.path.join(base_path, "data", "qsd1_w1")
bbdd = os.path.join(base_path, "data", "BBDD")

# Load GT (1D histograms computed for all images in the BBDD)
with open(qsd1 + "/gt_corresps.pkl", 'rb') as f:
	y = pickle.load(f)

	
def main():
	# Task 1: Create Museum and query image descriptors (BBDD & QSD1)
	image_filename = "00024.jpg"
	image_path = os.path.join(qsd1, image_filename)
	img = cv.imread(image_path)
	
	# Modify according to the method used
	color_space = "HSV"					# "Lab" or "HSV"
	k_value1 = 1							
	k_value2 = 5

	# Example CieLab histogram
	img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
	plot_hist_task1(img,image_filename, img_lab, 'Lab')

	# Example HSV histogram
	img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	plot_hist_task1(img,image_filename, img_hsv, 'HSV')

	# Task 2: : Implement / compute similarity measures to compare images
	# Task 3: Implement retrieval system (retrieve top K results)

	compute_descriptors(qsd1, color_space)

	with open(os.path.join(qsd1, color_space+'_histograms.pkl'), 'rb') as f:
		qsd1_histograms = pickle.load(f)

	with open(os.path.join(bbdd, color_space+'_histograms.pkl'), 'rb') as f:
		bbdd_histograms = pickle.load(f)

	
	if color_space == "Lab":

		# Method 1 - Lab - Hellinger Kernel
		res_m1_k1 = []
		res_m1_k5 = []
		for query_lab_img in qsd1_histograms:
			res_m1_k1.append(
				compute_similarities(query_hist=query_lab_img, bbdd_histograms=bbdd_histograms,
									 method=cv.HISTCMP_HELLINGER,
									 k=k_value1)[1])
			res_m1_k5.append((compute_similarities(query_hist=query_lab_img, bbdd_histograms=bbdd_histograms,
												   method=cv.HISTCMP_HELLINGER, k=k_value2)[1]))

		print(f"mAP@{k_value1} for Method 1 - LAB: {mapk(y, res_m1_k1, k_value1)}")
		print(f"mAP@{k_value2} for Method 1 - LAB: {mapk(y, res_m1_k5, k_value2)}")
	
	elif color_space == "HSV":

		# Method 2 - HSV - Hellinger Kernel
		res_m2_k1 = []
		res_m2_k5 = []
		for query_hsv_img in qsd1_histograms:
			res_m2_k1.append(
				compute_similarities(query_hist=query_hsv_img, bbdd_histograms=bbdd_histograms,
									method=cv.HISTCMP_CHISQR_ALT,
									k=k_value1)[1])
			res_m2_k5.append(
				compute_similarities(query_hist=query_hsv_img, bbdd_histograms=bbdd_histograms,
									method=cv.HISTCMP_CHISQR_ALT,
									k=k_value2)[1])

		print(f"mAP@{k_value1} for Method 2 - HSV: {mapk(y, res_m2_k1, k_value1)}")
		print(f"mAP@{k_value2} for Method 2 - HSV: {mapk(y, res_m2_k5, k_value2)}")

	# Task 4: Create predictions for blind challenge (QST1)


if __name__ == '__main__':
	main()
