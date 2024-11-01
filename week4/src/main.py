import pickle
import os
import cv2 as cv
import argparse
from tqdm import tqdm
from matplotlib.image import imread

from compute_similarities import compute_similarities
from average_precision import mapk
from metrics import Metrics
from keypoint_detection import *

# Get the path of the folder containing the museum dataset (BBDD)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bbdd_path = os.path.join(base_path, "data", "BBDD")


def main():

	# Read the arguments from the command line
	parser = argparse.ArgumentParser(description="Retrieve results and compute mAP@k")
	parser.add_argument("query_path", help="Path to the query dataset")
	parser.add_argument("--similarity_measure", help="Similarity Measure (e.g., HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT)", default="HISTCMP_HELLINGER")
	parser.add_argument("--k_value", help="Top k results", default=1)
	parser.add_argument("--descriptor_type", help ="Descriptor texture type")
	parser.add_argument("--is_test", help="True if we are testing the model (without ground truth)", default=False, type=bool)
	

	args = parser.parse_args()
	similarity_measure = args.similarity_measure
	k_value = int(args.k_value)
	q_path = os.path.join(base_path, args.query_path)
	is_test = args.is_test
	descriptor_type = args.descriptor_type
	

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
	descriptors = []

	convert_y = False

	for filename in tqdm(files, desc="Processing images", unit="image"):
		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(q_path, filename))

		if descriptor_type == 'sift':
			kp, des = sift(img_bgr)

		elif descriptor_type == 'orb':
			kp, des = orb(img_bgr)
			
		

		# Extract the main index (before the underscore) and subindex (after the underscore if it exists)
		if '_' in filename:
			convert_y = True
			main_index, sub_index = map(int, filename.split('.')[0].split('_'))

			# Ensure that the histograms array is large enough to accommodate the main index
			if len(descriptors) <= main_index:
				descriptors.extend([[] for _ in range(main_index + 1 - len(descriptors))])

			# Ensure that the sublist for the main index exists and extend it if necessary
			if len(descriptors[main_index]) <= sub_index:
				descriptors[main_index].extend([None] * (sub_index + 1 - len(descriptors[main_index])))

			# Store the histogram at the correct position
			descriptors[main_index][sub_index] = des
		else:
			index = int(filename.split('.')[0])

			if len(descriptors) <= index:
				descriptors.extend([None] * (index + 1 - len(descriptors)))
			descriptors[index] = des

	if convert_y and not is_test:
		y = [[[item] for item in sublist] for sublist in y]

	
	with open(os.path.join(q_path, f'{descriptor_type}_descriptors.pkl'), 'wb') as f:
		pickle.dump(descriptors, f)

		
	with open(os.path.join(q_path,  f'{descriptor_type}_descriptors.pkl'), 'rb') as f:
		query_histograms = pickle.load(f)

	with open(os.path.join(bbdd_path,  f'{descriptor_type}_descriptors.pkl'), 'rb') as f:
		bbdd_histograms = pickle.load(f)
	
	

	# For each image in the query set, compute its similarity to all museum images (BBDD).
	res_m = []
	for query_img_h in tqdm(query_histograms, desc="Processing images", unit="image"):
		if len(query_img_h) <= 2:
			res_m_sub = []
			for query_img_h_sub in query_img_h:
				res_m_sub.append(compute_similarities(query_img_h_sub, bbdd_histograms, similarity_function, k_value)[1])
			res_m.append(res_m_sub)
			continue
		res_m.append(compute_similarities(query_img_h, bbdd_histograms, similarity_function, k_value)[1])
	
	# If we are not in testing mode
	if not is_test:
		
		# Save the top K indices of the museum images with the best similarity for each query image to a pickle file
		with open(os.path.join(q_path, f'{descriptor_type}_descriptors_{str(k_value)}_results.pkl'), 'wb') as f:
			pickle.dump(res_m, f)

		# Evaluate the results using mAP@K if we are not in testing mode	
		print(f"mAP@{k_value} for {descriptor_type}: {mapk(y, res_m, k_value)}")

		
	# Save the 'blind' results for the test query set 
	if is_test:
		#subdirectory_path = os.path.join(q_path, color_space)
		#os.makedirs(subdirectory_path, exist_ok=True)
		output_file_path = os.path.join(q_path, 'result.pkl')
		with open(output_file_path, 'wb') as f:
			pickle.dump(res_m, f)

if __name__ == '__main__':
	main()
