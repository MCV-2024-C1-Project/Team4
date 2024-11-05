import pickle
import argparse
from tqdm import tqdm

from compute_similarities import compute_similarities_bidirectional, compute_similarities_daisy
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
	parser.add_argument("--k_value", help="Top k results", default=1)
	parser.add_argument("--descriptor_type", help="Descriptor type")
	parser.add_argument("--is_test", help="True if we are testing the model (without ground truth)", default=False, type=bool)
	

	args = parser.parse_args()
	k_value = int(args.k_value)
	q_path = os.path.join(base_path, args.query_path)
	is_test = args.is_test
	descriptor_type = args.descriptor_type

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

		# Resize the image to 256x256
		if descriptor_type == 'sift':
			img_bgr = cv.resize(img_bgr, (256, 256))
			kp, des = sift(img_bgr)

		elif descriptor_type == 'orb':
			img_bgr = cv.resize(img_bgr, (256, 256))
			kp, des = orb(img_bgr)

		elif descriptor_type == 'daisy':
			img_bgr = cv.resize(img_bgr, (256, 256), interpolation=cv.INTER_AREA)
			des, shape = daisy_descriptor(img_bgr)
			
		

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
		query_descriptors = pickle.load(f)

	with open(os.path.join(bbdd_path,  f'{descriptor_type}_descriptors.pkl'), 'rb') as f:
		bbdd_descriptors = pickle.load(f)
	
	

	# For each image in the query set, compute its similarity to all museum images (BBDD).
	res_m = []
	for query_img_h in tqdm(query_descriptors, desc="Processing images", unit="image"):
		if len(query_img_h) <= 2:
			res_m_sub = []
			for query_img_h_sub in query_img_h:
				if descriptor_type == 'daisy':
					res_m_sub.append(compute_similarities_daisy(query_img_h_sub, bbdd_descriptors, k_value)[1])
				else:
					res_m_sub.append(compute_similarities_bidirectional(query_img_h_sub, bbdd_descriptors, descriptor_type, k_value)[1])
			res_m.append(res_m_sub)
			continue
		if descriptor_type == 'daisy':
			res_m.append(compute_similarities_daisy(query_img_h, bbdd_descriptors, k_value)[1])
		else:
			res_m.append(compute_similarities_bidirectional(query_img_h, bbdd_descriptors, descriptor_type, k_value)[1])
	
	print(res_m)

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
