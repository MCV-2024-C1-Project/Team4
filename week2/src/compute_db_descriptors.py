import os
import cv2 as cv
import pickle

from compute_descriptors import compute_descriptors
from histograms import block_histogram

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
	# Compute descriptors for the BBDD images (offline)
	imgs_path = os.path.join(base_path, "data", "BBDD")
	files = [f for f in os.listdir(imgs_path) if f.endswith('.jpg')]
	files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
	color_space = "HSV"
	total_blocks = 4
	histograms = []
	for filename in files:

		# Read image (by default the color space of the loaded image is BGR) 
		img_bgr = cv.imread(os.path.join(imgs_path, filename))

		# Change color space (only 2 options are possible)
		if color_space == "Lab":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
			bins_channel1 = 256
			ranges = [0,256,0,256,0,256]
		elif color_space == "HSV":
			img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
			bins_channel1 = 32
			ranges = [0,180,0,256,0,256]

		hist = block_histogram(img,total_blocks,bins_channel1,ranges)

		index = int(filename.split('_')[-1].split('.')[0])

		if len(histograms) <= index:
			histograms.extend([None] * (index + 1 - len(histograms)))
		histograms[index] = hist

	
	with open(os.path.join(imgs_path, color_space + '_histograms.pkl'), 'wb') as f:
		pickle.dump(histograms, f)
	#compute_descriptors(imgs_path, color_space="Lab")
	#compute_descriptors(imgs_path, color_space="HSV")


if __name__ == "__main__":
	main()
