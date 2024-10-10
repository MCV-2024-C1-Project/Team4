import cv2 as cv
import argparse
import numpy as np


def fill_surrounded_pixels(foreground):
	h, w = foreground.shape
	new_mask = foreground.copy()

	for i in range(h):
		for j in range(w):
			if foreground[i, j] == 0:
				# Check if there is a 1 above
				has_one_above = np.any(foreground[:i, j] == 1) if i > 0 else False
				# Check if there is a 1 below
				has_one_below = np.any(foreground[i + 1:, j] == 1) if i < h - 1 else False
				# Check if there is a 1 to the left
				has_one_left = np.any(foreground[i, :j] == 1) if j > 0 else False
				# Check if there is a 1 to the right
				has_one_right = np.any(foreground[i, j + 1:] == 1) if j < w - 1 else False

				# If there is at least a 1 in each direction, change the pixel to 1
				if has_one_above and has_one_below and has_one_left and has_one_right:
					new_mask[i, j] = 1
	return new_mask


def main():
	# Get the image path argument
	#parser = argparse.ArgumentParser(description="Remove background from an image")
	#parser.add_argument("image_path", help="Path to the image")
	#args = parser.parse_args()

	# Read image
	image = cv.imread("D:\Team4\data\qsd2_w1\\00002.jpg")

	# Show the original image
	cv.imshow("Original Image", image)
	cv.waitKey(0)

	myimage_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	# Take S and remove any value that is less than half
	s = myimage_hsv[:, :, 1]
	s = np.where(s < (255 * 0.45), 0, 1)  # Any value below 127 will be excluded

	# We increase the brightness of the image and then mod by 255
	v = myimage_hsv[:, :, 2]
	v = np.where(v > (255 * 0.35), 0, 1)  # Any value above 255 will be part of our mask

	# Combine our two masks based on S and V into a single "Foreground"
	foreground = np.where((s == 0) & (v == 0), 0, 1).astype(np.uint8)
	foreground[:10, :] = 0; foreground[-10:, :] = 0; foreground[:, :10] = 0; foreground[:, -10:] = 0
	foreground = fill_surrounded_pixels(foreground)
	cv.imshow("Foreground Mask", foreground * 255)
	cv.waitKey(0)

	kernel = np.ones((5, 5), np.uint8)
	opening = cv.morphologyEx(foreground, cv.MORPH_OPEN, kernel)
	cv.imshow("Foreground Mask with Opening", opening * 255)
	cv.waitKey(0)
	kernel = np.ones((50, 50), np.uint8)
	foreground = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
	cv.imshow("Foreground Mask with Opening/Closing", foreground * 255)
	cv.waitKey(0)

	# Show background mask
	background = np.where(foreground == 0, 255, 0).astype(np.uint8)  # Invert foreground to get background in uint8
	background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)  # Convert background back into BGR space
	cv.imshow("Background Mask", background)
	cv.waitKey(0)

	foreground = cv.bitwise_and(image, image, mask=foreground)  # Apply our foreground map to original image

	# Show the final image
	finalimage = foreground - background  # Combine foreground and background
	cv.imshow("Final Image", finalimage)
	cv.waitKey(0)


if __name__ == "__main__":
	main()
