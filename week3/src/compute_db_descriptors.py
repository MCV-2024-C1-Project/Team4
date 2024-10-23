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

from texture_descriptors import *

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

	# Argument parser
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--num_blocks', type=int, help='Number of blocks for block histogram', default=4)
	parser.add_argument('--num_bins', type=int, help='Number of bins for histogram', default=16)
	parser.add_argument('--num_levels', type=int, help='Number of wavelet decomposition levels')
	parser.add_argument('--descriptor_type')
	parser.add_argument('--wavelet_type', help='Type of wavelet to use (db1, haar)')
	parser.add_argument('--N', help='Number of DCT coefficients to keep', default=None)
	args = parser.parse_args()

	NUM_BLOCKS = int(args.num_blocks) if args.num_blocks else None
	NUM_BINS = int(args.num_bins) if args.num_bins else None
	DESCRIPTOR_TYPE = args.descriptor_type
	NUM_LEVELS = int(args.num_levels) if args.num_levels else None
	WAVELET_TYPE = args.wavelet_type
	

	# Compute descriptors for the BBDD images (offline)
	imgs_path = os.path.join(base_path, "data", "BBDD")
	# Get all the images of the BBDD that have extension '.jpg'
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

	histograms = []
	for filename in tqdm(files, desc="Processing images", unit="image"):
		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(imgs_path, filename))

		if DESCRIPTOR_TYPE == 'LBP':
			hist = lbp_block_histogram(img_bgr,total_blocks = NUM_BLOCKS, bins = NUM_BINS)

		elif DESCRIPTOR_TYPE == 'DCT':
			hist = dct_block_histogram(img_bgr, total_blocks=NUM_BLOCKS, bins=NUM_BINS, N=int(args.N))
		
		elif DESCRIPTOR_TYPE == 'wavelet':
			A = imread(os.path.join(imgs_path, filename))
			hist = wavelet_descriptor(A, wavelet=WAVELET_TYPE, level=NUM_LEVELS)
		
		# TODO: Add more texture descriptors here

		index = int(filename.split('_')[-1].split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

	if DESCRIPTOR_TYPE == 'wavelet':
		with open(os.path.join(imgs_path, f'{DESCRIPTOR_TYPE}_histograms_{WAVELET_TYPE}_type_{NUM_LEVELS}_levels.pkl'), 'wb') as f:
			pickle.dump(histograms, f)
	elif DESCRIPTOR_TYPE == 'DCT':
		filename = f'{DESCRIPTOR_TYPE}_{NUM_BLOCKS}_blocks_{NUM_BINS}_bins.pkl'
		if args.N is not None:
			filename = f'{DESCRIPTOR_TYPE}_{NUM_BLOCKS}_blocks_{args.N}_coefficients.pkl'
		with open(os.path.join(imgs_path, filename), 'wb') as f:
			pickle.dump(histograms, f)
	else:
		with open(os.path.join(imgs_path, f'{DESCRIPTOR_TYPE}_histograms_{NUM_BLOCKS}_blocks_{NUM_BINS}_bins.pkl'), 'wb') as f:
			pickle.dump(histograms, f)

if __name__ == "__main__":
	main()
