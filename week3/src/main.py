import pickle
import os
import cv2 as cv
import argparse
from tqdm import tqdm
from matplotlib.image import imread

from compute_similarities import compute_similarities
from average_precision import mapk
from metrics import Metrics
from texture_descriptors import *

# Get the path of the folder containing the museum dataset (BBDD)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bbdd_path = os.path.join(base_path, "data", "BBDD")


def main():

	# Read the arguments from the command line
	parser = argparse.ArgumentParser(description="Retrieve results and compute mAP@k")
	parser.add_argument("query_path", help="Path to the query dataset")
	parser.add_argument("--num_blocks", help="Number of blocks fot the block histogram", default=1)
	parser.add_argument("--similarity_measure", help="Similarity Measure (e.g., HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT)", default="HISTCMP_HELLINGER")
	parser.add_argument("--num_bins", help="Number of bins for the histogram", default=16)
	parser.add_argument('--num_levels')
	parser.add_argument("--k_value", help="Top k results", default=1)
	parser.add_argument("--descriptor_type", help ="Descriptor texture type")
	parser.add_argument("--is_test", help="True if we are testing the model (without ground truth)", default=False, type=bool)
	parser.add_argument('--wavelet_type', help='Type of wavelet to use (db1, haar)')

	args = parser.parse_args()
	num_blocks = int(args.num_blocks)
	num_bins = int(args.num_bins)
	similarity_measure = args.similarity_measure
	k_value = int(args.k_value)
	q_path = os.path.join(base_path, args.query_path)
	is_test = args.is_test
	descriptor_type = args.descriptor_type
	num_levels = int(args.num_levels)
	wavelet_type = args.wavelet_type

	# Select the appropriate similarity measure based on the command line argument. 
	# For those distances that we have defined manually, we have assigned
	# them a numerical ID.
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
	elif similarity_measure == "Ssim":
		similarity_function = Metrics.SSIM
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

		if descriptor_type == 'LBP':
			hist = lbp_block_histogram(img_bgr,total_blocks = num_blocks,bins = num_bins)

		elif descriptor_type == 'DCT':
			hist = dct_block_histogram(img_bgr, total_blocks=num_blocks, bins=num_bins)
		
		elif descriptor_type == 'wavelet':
			A = imread(os.path.join(q_path, filename))
			hist = wavelet_descriptor(A, wavelet=wavelet_type, level=num_levels)

		# TODO: Add more texture descriptors here

		# Extract the index (numerical ID) from the filename and store histogram at the correct position
		index = int(filename.split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

	if descriptor_type == 'wavelet':
		# Save query histograms to a pickle file
		with open(os.path.join(q_path, f'{descriptor_type}_histograms_{wavelet_type}_type_{num_levels}_levels.pkl'), 'wb') as f:
			pickle.dump(histograms, f)

		# Load the precomputed image descriptors from '.pkl' files
		# for both the query dataset  and the museum dataset (BBDD, computed offline)
		with open(os.path.join(q_path,  f'{descriptor_type}_histograms_{wavelet_type}_type_{num_levels}_levels.pkl'), 'rb') as f:
			query_histograms = pickle.load(f)

		with open(os.path.join(bbdd_path,  f'{descriptor_type}_histograms_{wavelet_type}_type_{num_levels}_levels.pkl'), 'rb') as f:
			bbdd_histograms = pickle.load(f)

	
	else:
		# Save query histograms to a pickle file
		with open(os.path.join(q_path, f'{descriptor_type}_histograms_{num_blocks}_blocks_{num_bins}_bins.pkl'), 'wb') as f:
			pickle.dump(histograms, f)

		# Load the precomputed image descriptors from '.pkl' files
		# for both the query dataset  and the museum dataset (BBDD, computed offline)
		with open(os.path.join(q_path,  f'{descriptor_type}_histograms_{num_blocks}_blocks_{num_bins}_bins.pkl'), 'rb') as f:
			query_histograms = pickle.load(f)

		with open(os.path.join(bbdd_path,  f'{descriptor_type}_histograms_{num_blocks}_blocks_{num_bins}_bins.pkl'), 'rb') as f:
			bbdd_histograms = pickle.load(f)
	

	# For each image in the query set, compute its similarity to all museum images (BBDD).
	res_m = []
	
	for query_img_h in tqdm(query_histograms, desc="Processing images", unit="image"):
		res_m.append(compute_similarities(query_img_h, bbdd_histograms, similarity_function, k_value)[1])
	
	# If we are not in testing mode
	if not is_test:
		if descriptor_type == 'wavelet':
			# Save the top K indices of the museum images with the best similarity for each query image to a pickle file
			with open(os.path.join(q_path, f'{descriptor_type}_{num_levels}_levels_{similarity_measure}_{wavelet_type}_type_{str(k_value)}_results.pkl'), 'wb') as f:
				pickle.dump(res_m, f)

			# Evaluate the results using mAP@K if we are not in testing mode	
			print(f"mAP@{k_value} for {descriptor_type}: {mapk(y, res_m, k_value)}")

		else:
			# Save the top K indices of the museum images with the best similarity for each query image to a pickle file
			with open(os.path.join(q_path, f'{descriptor_type}_{num_blocks}_blocks_{num_bins}_bins_{similarity_measure}_{str(k_value)}_results.pkl'), 'wb') as f:
				pickle.dump(res_m, f)

			# Evaluate the results using mAP@K if we are not in testing mode	
			print(f"mAP@{k_value} for {descriptor_type}: {mapk(y, res_m, k_value)}")

	'''
	# Save the 'blind' results for the test query set 
	if is_test:
		subdirectory_path = os.path.join(q_path, color_space)
		os.makedirs(subdirectory_path, exist_ok=True)
		output_file_path = os.path.join(subdirectory_path, 'result.pkl')
		with open(output_file_path, 'wb') as f:
			pickle.dump(res_m, f)
	'''

if __name__ == '__main__':
	main()
