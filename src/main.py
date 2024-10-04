import pickle
import os
import cv2 as cv
import argparse

from compute_similarities import compute_similarities
from compute_descriptors import compute_descriptors
from average_precision import mapk
from utils import plot_hist_task1

# Get the paths of the folders containing the query set development (QST1), 
# the paintings dataset (BBDD), and the query set test (QST1)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bbdd_path = os.path.join(base_path, "data", "BBDD")
	
def main():

	# Read the color space and k value from the command line
	parser = argparse.ArgumentParser(description="Retrieve results and compute mAP@k")
	parser.add_argument("color_space", help="Color space (e.g., Lab or HSV)")
	parser.add_argument("k_value", help="Top k results")
	parser.add_argument("query_path", help="Path to the query dataset")
	parser.add_argument("is_test", help="True if we are testing the model (without ground truth)")

	args = parser.parse_args()
	color_space = args.color_space
	k_value = int(args.k_value)
	query_path = args.query_path
	is_test = args.is_test == "True"

	if not is_test:
		with open(query_path + "/gt_corresps.pkl", 'rb') as f:
			y = pickle.load(f)

	# Task 1: Create Museum and query image descriptors (BBDD & QSD1)
	#	   -> Example of creted image descriptors: For one painting in 
	# 		  the QST1 we read its '.jpg' file (BGR image). 
	image_filename = "00024.jpg"
	image_path = os.path.join(query_path, image_filename)
	img = cv.imread(image_path)

	#	   -> Method1: We compute and plot the color histogram of the 
	# 		  chosen query painting and plot it. The color space for
	#		  the 1st method is CieLab
	img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
	plot_hist_task1(img,image_filename, img_lab, 'Lab')

	#	   -> Method2: The color space for the 2nd method is HSV
	img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	plot_hist_task1(img,image_filename, img_hsv, 'HSV')

	#	   -> We compute image descriptors for all images in QST1 
	#		  (are saved in a '.pkl' file) according to the color 
	# 		  space specified as an argument when running the main()  
	compute_descriptors(query_path, color_space)


	# Task 2: Implement / compute similarity measures to compare images
	#	   -> Selection of similarity measures is discussed in the README.md file	


	# Task 3: Implement retrieval system (retrieve top K results)
	#	   -> We read the '.pkl' files conatining the computed image 
	# 		  descriptors of the query dataset (QST1) and the museum 
	# 		  dataset (BBDD, computed offline) 
	with open(os.path.join(query_path, color_space+'_histograms.pkl'), 'rb') as f:
		query_histograms = pickle.load(f)

	with open(os.path.join(bbdd_path, color_space+'_histograms.pkl'), 'rb') as f:
		bbdd_histograms = pickle.load(f)

	#	   -> For each image in the query set (QST1), we compute its 
	# 		  similarity to all museum images (BBDD) using the similarity 
	# 		  measures chosen in Task2. 
	#			- Method 1 (Lab): Hellinger (distance)
	#			- Method 2 (HSV): Alternative Chi Square (distance)
	res_m = []
	if color_space == "Lab":
		for query_lab_img in query_histograms:
			res_m.append(
				compute_similarities(query_hist=query_lab_img, bbdd_histograms=bbdd_histograms,
									 method=cv.HISTCMP_HELLINGER, k=k_value)[1])
	
	elif color_space == "HSV":
		for query_hsv_img in query_histograms:
			res_m.append(
				compute_similarities(query_hist=query_hsv_img, bbdd_histograms=bbdd_histograms,
									method=cv.HISTCMP_CHISQR_ALT, k=k_value)[1])

	if not is_test:
		print(f"mAP@{k_value} for {color_space}: {mapk(y, res_m, k_value)}")

	# Task 4: Create predictions for blind challenge (QST1)
	if is_test:
		subdirectory_path = os.path.join(query_path, color_space)
		os.makedirs(subdirectory_path, exist_ok=True)
		output_file_path = os.path.join(subdirectory_path, 'result.pkl')
		with open(output_file_path, 'wb') as f:
			pickle.dump(res_m, f)

if __name__ == '__main__':
	main()
