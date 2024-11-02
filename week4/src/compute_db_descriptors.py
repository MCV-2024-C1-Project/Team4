"""
Compute the 3D Histograms for each image in the BBDD. Each 3D histogram is saved
as a list of length n_bins³ (assuming n_bins is the same for the three channels).
Histograms are saved in a .pkl file.
"""
import os
import cv2 as cv
import pickle
import argparse
from tqdm import tqdm
from matplotlib.image import imread

from keypoint_detection import *

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

	# Argument parser
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--descriptor_type')
	args = parser.parse_args()

	DESCRIPTOR_TYPE = args.descriptor_type

	

	# Compute descriptors for the BBDD images (offline)
	imgs_path = os.path.join(base_path, "data", "BBDD")
	# Get all the images of the BBDD that have extension '.jpg'
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

	descriptors = []
	for filename in tqdm(files, desc="Processing images", unit="image"):
		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(imgs_path, filename))

		if DESCRIPTOR_TYPE == 'sift':
			kp, des = sift(img_bgr)
		elif DESCRIPTOR_TYPE == 'orb':
			kp, des = orb(img_bgr)
		elif DESCRIPTOR_TYPE == 'daisy':
			des = daisy_descriptor(img_bgr)
			des = des.astype(np.float32)
			

		index = int(filename.split('_')[-1].split('.')[0])

		if len(descriptors) <= index:
			descriptors.extend([None] * (index + 1 - len(descriptors)))
		descriptors[index] = des

	
	with open(os.path.join(imgs_path, f'{DESCRIPTOR_TYPE}_descriptors.pkl'), 'wb') as f:
		pickle.dump(descriptors, f)
	

		

if __name__ == "__main__":
	main()
