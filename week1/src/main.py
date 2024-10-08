import pickle
import os
import cv2 as cv
import argparse

from compute_similarities import compute_similarities
from compute_descriptors import compute_descriptors
from average_precision import mapk
from metrics import Metrics

# Get the path of the folder containing the museum dataset (BBDD)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bbdd_path = os.path.join(base_path, "data", "BBDD")


def main():

	# Read the color space, similarity measure, k value, query dataset path, and test flag from the command line
	parser = argparse.ArgumentParser(description="Retrieve results and compute mAP@k")
	parser.add_argument("color_space", help="Color space (e.g., Lab, HSV)")
	parser.add_argument("similarity_measure", help="Similarity Measure (e.g., HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT)")
	parser.add_argument("k_value", help="Top k results")
	parser.add_argument("query_path", help="Path to the query dataset")
	parser.add_argument("is_test", help="True if we are testing the model (without ground truth)")

	args = parser.parse_args()
	color_space = args.color_space
	similarity_measure = args.similarity_measure
	k_value = int(args.k_value)
	q_path = os.path.join(base_path, args.query_path)
	is_test = args.is_test == "True"

	# Select the appropriate similarity measure based on the command line argument. 
	# For those distances that we have defined manually, we have assigned them a numerical ID.
	if similarity_measure == "HISTCMP_CORREL":
		similarity_function = cv.HISTCMP_CORREL
	elif similarity_measure == "HISTCMP_CHISQR":
		similarity_function = cv.HISTCMP_CHISQR
	elif similarity_measure == "HISTCMP_INTERSECT":
		similarity_function = cv.HISTCMP_INTERSECT
	elif similarity_measure == "HISTCMP_BHATTACHARYYA":
		similarity_function = cv.HISTCMP_BHATTACHARYYA
	elif similarity_measure == "HISTCMP_HELLINGER":
		similarity_function = cv.HISTCMP_HELLINGER
	elif similarity_measure == "HISTCMP_CHISQR_ALT":
		similarity_function = cv.HISTCMP_CHISQR_ALT
	elif similarity_measure == "HISTCMP_KL_DIV":
		similarity_function = cv.HISTCMP_KL_DIV
	elif similarity_measure == "Manhattan":
		similarity_function = Metrics.MANHATTAN
	elif similarity_measure == "Lorentzian":
		similarity_function = Metrics.LORENTZIAN
	elif similarity_measure == "Canberra":
		similarity_function = Metrics.CANBERRA
	else:
		raise ValueError(f"Unknown similarity measure: {similarity_measure}")

	## Task 3: Implement retrieval system (retrieve top K results)

	# If we are not testing, we get the provided GT to evaluate the results 
	# obatined for the QSD1
	if not is_test:
		with open(q_path + "/gt_corresps.pkl", 'rb') as f:
			y = pickle.load(f) 

	# Compute image descriptors for all images in the query dataset (QST1) 
	# and save them in a '.pkl' file based on the specified color space
	compute_descriptors(q_path, color_space)
	
	# Load the precomputed image descriptors from '.pkl' files
	# for both the query dataset (QST1) and the museum dataset (BBDD, computed offline)
	with open(os.path.join(q_path, color_space+'_histograms.pkl'), 'rb') as f:
		query_histograms = pickle.load(f)

	with open(os.path.join(bbdd_path, color_space+'_histograms.pkl'), 'rb') as f:
		bbdd_histograms = pickle.load(f)

	# For each image in the query set (QST1), compute its similarity to all museum images (BBDD).
	# The best results are obtained using the following similarity measures:	  
	#	- Method 1 (Lab): Hellinger (distance)
	#	- Method 2 (HSV): Alternative Chi Square (distance)
	res_m = []
	
	for query_img_h in query_histograms:
		res_m.append(compute_similarities(query_img_h, bbdd_histograms, similarity_function, k_value)[1])
	
	# If we are not in testing mode
	if not is_test:
		# Save the top K indices of the museum images with the best similarity for each query image to a pickle file
		with open(os.path.join(q_path, color_space +'_'+ similarity_measure + '_' + str(k_value) + '_results.pkl'), 'wb') as f:
			pickle.dump(res_m, f)

		# Evaluate the results using mAP@K if we are not in testing mode	
		print(f"mAP@{k_value} for {color_space}: {mapk(y, res_m, k_value)}")

	## Task 4: Create predictions for blind challenge (QST1)

	# Save the 'blind' results for the test query set 
	if is_test:
		subdirectory_path = os.path.join(q_path, color_space)
		os.makedirs(subdirectory_path, exist_ok=True)
		output_file_path = os.path.join(subdirectory_path, 'result.pkl')
		with open(output_file_path, 'wb') as f:
			pickle.dump(res_m, f)

if __name__ == '__main__':
	main()
