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

from texture_descriptors import lbp_block_histogram

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

	# Argument parser
	parser = argparse.ArgumentParser(description="Compute 3D histograms for each image in the BBDD")
	parser.add_argument('--color_space', type=str, choices=['Lab', 'HSV', 'RGB', 'HLS', 'Luv', 'YCrCb', 'YUV'], help='Color space to use')
	parser.add_argument('--num_blocks', type=int, help='Number of blocks for block histogram')
	parser.add_argument('--num_levels', type=int, help='Number of levels for block histogram')
	parser.add_argument('--num_bins', type=int, help='Number of bins for histogram')
	parser.add_argument("--is_pyramid", help="True if we are using the spatial pyramid histogram mode", default=False, type=bool)
	args = parser.parse_args()

	COLOR_SPACE = args.color_space
	NUM_BLOCKS = args.num_blocks
	NUM_LEVELS = args.num_levels
	NUM_BINS = args.num_bins
	is_pyramid = args.is_pyramid

	# Compute descriptors for the BBDD images (offline)
	imgs_path = os.path.join(base_path, "data", "BBDD")
	# Get all the images of the BBDD that have extension '.jpg'
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

	histograms = []
	for filename in tqdm(files, desc="Processing images", unit="image"):
		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(imgs_path, filename))

		# Change color space (only 2 options are possible)
		if COLOR_SPACE == "Lab":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
			ranges = [0,256,0,256,0,256]
		elif COLOR_SPACE == "HSV":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
			ranges = [0,180,0,256,0,256]
		elif COLOR_SPACE == "RGB":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
			ranges = [0,256,0,256,0,256]
		elif COLOR_SPACE == "HLS":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HLS)
			ranges = [0,256,0,256,0,256]
		elif COLOR_SPACE == "Luv":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Luv)
			ranges = [0,256,0,256,0,256]
		elif COLOR_SPACE == "YCrCb":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
			ranges = [0,256,0,256,0,256]
		elif COLOR_SPACE == "YUV":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2YUV)
			ranges = [0,256,0,256,0,256]

		# Compute the 3D histogram

		if not is_pyramid:
			# Compute the 3D Block Histograms for the query image
			hist = block_histogram(img,NUM_BLOCKS,NUM_BINS,ranges)
		else:
			# Compute the 3D Hierarchical Histograms for the query image
			hist = spatial_pyramid_histogram(img, NUM_LEVELS, NUM_BINS, ranges)

		index = int(filename.split('_')[-1].split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

	if not is_pyramid:
		with open(os.path.join(imgs_path, COLOR_SPACE + '_histograms_'+str(NUM_BLOCKS)+'_blocks_'+str(NUM_BINS)+'_bins'+'.pkl'), 'wb') as f:
			pickle.dump(histograms, f)
	else:
		with open(os.path.join(imgs_path, COLOR_SPACE + '_histograms_'+str(NUM_LEVELS)+'_levels_'+str(NUM_BINS)+'_bins'+'.pkl'), 'wb') as f:
			pickle.dump(histograms, f)


if __name__ == "__main__":
	main()
