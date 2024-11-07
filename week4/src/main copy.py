import pickle
import argparse
from tqdm import tqdm

from compute_similarities import *
from average_precision import mapk
from metrics import Metrics
from keypoint_detection import *

# Get the path of the folder containing the museum dataset (BBDD)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bbdd_path = os.path.join(base_path, "data", "BBDD")


def main():

	# Read the arguments from the command line
	k_value = 1
	q_path = os.path.join(base_path, "data/qsd1_w4/images_without_noise/masked")
	is_test = False
	descriptor_type = "daisy"

	# If we are not testing, we get the provided GT to evaluate the results 
	# obatined for the QSD1
	if not is_test:
		with open(q_path + "/gt_corresps.pkl", 'rb') as f:
			y = pickle.load(f) 

	'''
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
		elif descriptor_type == 'daisy':
			des = daisy_descriptor(img_bgr)
			des = des.astype(np.float32)
			
		

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
	'''
		
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
				res_m_sub.append(compute_similarities_daisy2(query_img_h_sub, bbdd_histograms, k_value)[1])
			res_m.append(res_m_sub)
			continue
		res_m.append(compute_similarities_daisy2(query_img_h, bbdd_histograms, k_value)[1])
	print(res_m)
	
	
	# If we are not in testing mode
	if not is_test:
		
		# Save the top K indices of the museum images with the best similarity for each query image to a pickle file
		with open(os.path.join(q_path, f'{descriptor_type}_descriptors_{str(k_value)}_results.pkl'), 'wb') as f:
			pickle.dump(res_m, f)

		# Evaluate the results using mAP@K if we are not in testing mode	
		print(f"mAP@{k_value} for {descriptor_type}: {mapk(y, res_m, k_value)}")


if __name__ == '__main__':
	main()
