import pickle
import os
import cv2 as cv
import argparse
from tqdm import tqdm

from compute_similarities import compute_similarities
from histograms import block_histogram, spatial_pyramid_histogram
from average_precision import mapk
from metrics import Metrics

# Get the path of the folder containing the museum dataset (BBDD)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bbdd_path = os.path.join(base_path, "data", "BBDD")


def main():

	# Read the arguments from the command line
	parser = argparse.ArgumentParser(description="Retrieve results and compute mAP@k")
	parser.add_argument("query_path", help="Path to the query dataset")
	parser.add_argument("--color_space", help="Color space (e.g., Lab, HSV)", default="Lab")
	parser.add_argument("--num_blocks", help="Number of blocks or levels", default=1)
	parser.add_argument("--num_bins", help="Number of bins (same for all channels) to compute histograms", default=1)
	parser.add_argument("--similarity_measure", help="Similarity Measure (e.g., HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT)", default="HISTCMP_HELLINGER")
	parser.add_argument("--k_value", help="Top k results", default=1)
	parser.add_argument("--is_pyramid", help="True if we are using the spatial pyramid histogram mode", default=False, type=bool)
	parser.add_argument("--is_test", help="True if we are testing the model (without ground truth)", default=False, type=bool)

	args = parser.parse_args()
	color_space = args.color_space
	num_blocks = int(args.num_blocks)
	num_bins = int(args.num_bins)
	is_pyramid = args.is_pyramid
	similarity_measure = args.similarity_measure
	k_value = int(args.k_value)
	q_path = os.path.join(base_path, args.query_path)
	is_test = args.is_test

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

	# If we are not testing, we get the provided GT to evaluate the results 
	# obatined for the QSD1
	if not is_test:
		with open(q_path + "/gt_corresps.pkl", 'rb') as f:
			y = pickle.load(f) 

	# Get all the images of the QSD that have extension '.jpg'
	files = [f for f in os.listdir(q_path) if f.endswith('.jpg')]
	histograms = []
	for filename in tqdm(files, desc="Processing images", unit="image"):
		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(q_path, filename))

		# Change color space (only 2 options are possible)
		if color_space == "Lab":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
			ranges = [0,256,0,256,0,256]
		elif color_space == "HSV":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
			ranges = [0,180,0,256,0,256]
		elif color_space == "RGB":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
			ranges = [0,256,0,256,0,256]
		elif color_space == "HLS":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HLS)
			ranges = [0,256,0,256,0,256]
		elif color_space == "Luv":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Luv)
			ranges = [0,256,0,256,0,256]
		elif color_space == "YCrCb":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
			ranges = [0,256,0,256,0,256]
		elif color_space == "YUV":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2YUV)
			ranges = [0,256,0,256,0,256]

		if not is_pyramid:
			# Compute the 3D Block Histograms for the query image
			hist = block_histogram(img,num_blocks,num_bins,ranges)
		else:
			# Compute the 3D Hierarchical Histograms for the query image
			num_levels = num_blocks
			hist = spatial_pyramid_histogram(img, num_levels, num_bins, ranges)

		# Extract the index (numerical ID) from the filename and store histogram at the correct position
		index = int(filename.split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

	if not is_pyramid:
		# Save query histograms to a pickle file
		with open(os.path.join(q_path, color_space + '_histograms_'+str(num_blocks)+'_blocks_'+str(num_bins)+'_bins'+'.pkl'), 'wb') as f:
			pickle.dump(histograms, f)

		# Load the precomputed image descriptors from '.pkl' files
		# for both the query dataset (QST1) and the museum dataset (BBDD, computed offline)
		with open(os.path.join(q_path, color_space + '_histograms_'+str(num_blocks)+'_blocks_'+str(num_bins)+'_bins'+'.pkl'), 'rb') as f:
			query_histograms = pickle.load(f)

		with open(os.path.join(bbdd_path, color_space + '_histograms_'+str(num_blocks)+'_blocks_'+str(num_bins)+'_bins'+'.pkl'), 'rb') as f:
			bbdd_histograms = pickle.load(f)
	else:
		with open(os.path.join(q_path, color_space + '_histograms_'+str(num_blocks)+'_levels_'+str(num_bins)+'_bins'+'.pkl'), 'wb') as f:
			pickle.dump(histograms, f)

		# Load the precomputed image descriptors from '.pkl' files
		# for both the query dataset (QST1) and the museum dataset (BBDD, computed offline)
		with open(os.path.join(q_path, color_space + '_histograms_'+str(num_blocks)+'_levels_'+str(num_bins)+'_bins'+'.pkl'), 'rb') as f:
			query_histograms = pickle.load(f)

		with open(os.path.join(bbdd_path, color_space + '_histograms_'+str(num_blocks)+'_levels_'+str(num_bins)+'_bins'+'.pkl'), 'rb') as f:
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
