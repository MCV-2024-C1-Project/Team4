import cv2 as cv
import argparse
import numpy as np
import os
import tqdm

from week3.src.metrics import global_f1_score
from week3.utils.utils import plot_mask_with_points, order_points


def fill_surrounded_pixels(foreground):
	"""
	Fill the surrounded pixels in the foreground
	:param foreground: The foreground mask
	:return: The new mask with surrounded pixels filled
	"""
	h, w = foreground.shape
	new_mask = foreground.copy()

	# Cumulative sums to check for foreground presence in each direction
	has_one_above = np.cumsum(foreground, axis=0) > 0
	has_one_below = np.cumsum(foreground[::-1, :], axis=0)[::-1] > 0
	has_one_left = np.cumsum(foreground, axis=1) > 0
	has_one_right = np.cumsum(foreground[:, ::-1], axis=1)[:, ::-1] > 0

	# Find all positions that have a 0 but are surrounded by 1s in all directions
	surrounded = (foreground == 0) & has_one_above & has_one_below & has_one_left & has_one_right

	# Update the new mask with 1 where pixels are surrounded
	new_mask[surrounded] = 1

	return new_mask


def get_extreme_points(mask):
	# Step 1: Find contours
	contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# Step 2: Get the largest contour (assuming the mask has one main object)
	contour = max(contours, key=cv.contourArea)

	# Step 3: Approximate the contour to a polygon (epsilon can be adjusted)
	epsilon = 0.02 * cv.arcLength(contour, True)
	approx = cv.approxPolyDP(contour, epsilon, True)

	# Step 4: If approxPolyDP does not return 4 points, we use cv2.goodFeaturesToTrack to get corners
	if len(approx) != 4:
		# Use corner detection if polygon approximation didn't give us 4 corners
		corners = cv.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=10)
		corners = corners.astype(int)
	else:
		corners = approx.reshape(-1, 2)

	# Step 5: Sort the corners based on their positions
	ordered_corners = order_points(corners)
	ordered_corners = ordered_corners.astype(int)

	# (Optional) Uncomment to display the mask with the points
	# plot_mask_with_points(mask, ordered_corners)

	return ordered_corners


def remove_background(image_path):
	"""
	Remove the background from an image
	:param image_path: Path to the image
	:return: The image with the background removed and the mask
	"""
	# Read image
	image = cv.imread(image_path)
	image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	# Create an empty mask
	mask = np.zeros(image.shape[:2], np.uint8)

	# Define background and foreground models
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)

	# Define rectangle (x, y, width, height) around the artwork
	height, width = image.shape[:2]
	rect = (10, 10, width - 20, height - 20)  # Adjust as needed

	# Apply GrabCut
	cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

	# Create mask where sure and likely foreground are 1
	mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')

	# Opening + Closing to remove noise
	kernel = np.ones((3, 3), np.uint8)
	mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
	foreground = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)

	# Extract foreground
	# foreground = image * mask2[:, :, np.newaxis]

	cv.imshow('Foreground', foreground * 255)
	cv.waitKey(0)

	foreground = fill_surrounded_pixels(foreground)

	# TODO: Update this part to allow multiple artworks in the image
	# foreground = apply_square_mask(foreground)

	# Find the bounding box of the non-background region
	height, width = foreground.shape[:2]
	pts1 = np.array(get_extreme_points(foreground), dtype="float32")
	pts2 = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

	M = cv.getPerspectiveTransform(pts1, pts2)
	dst = cv.warpPerspective(image, M, (width, height))
	dst = cv.cvtColor(dst, cv.COLOR_HSV2BGR)

	cv.imshow('Cropped Image', dst)
	cv.waitKey(0)

	cv.imshow('Foreground', foreground * 255)
	cv.waitKey(0)
	return dst, foreground


def load_masks(imgs_path):
	ground_truths = {}
	predicted_masks = {}

	# Load ground truth masks into a dictionary
	for filename in os.listdir(imgs_path):
		if filename.endswith(".png") and not filename.endswith("_mask.png"):
			mask_path = os.path.join(imgs_path, filename)
			mask = cv.imread(mask_path)  # Load in grayscale if needed
			key = filename.split(".")[0]  # Get the base name without extension
			ground_truths[key] = mask

	# Load predicted masks into a dictionary
	for filename in os.listdir(imgs_path):
		if filename.endswith("_mask.png"):
			mask_path = os.path.join(imgs_path, filename)
			mask = cv.imread(mask_path)
			key = filename.split("_mask")[0]  # Get the base name without the "_mask" suffix
			predicted_masks[key] = mask

	# Create aligned lists for ground truth and predicted masks
	ground_truth_list = []
	predicted_list = []
	for key in sorted(ground_truths.keys()):
		if key in predicted_masks:
			ground_truth_list.append(ground_truths[key])
			predicted_list.append(predicted_masks[key])
		else:
			print(f"Warning: No predicted mask found for ground truth {key}")

	return ground_truth_list, predicted_list


def main():
	# Get the image path argument
	parser = argparse.ArgumentParser(description="Remove background from an images")
	parser.add_argument("imgs_path", help="Path to the image folder")
	parser.add_argument("--score", help="Show F1 score, precision and recall", type=bool)
	args = parser.parse_args()

	base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	imgs_path = os.path.join(base_path, args.imgs_path)

	if not args.score:
		# Loop through all files in the directory
		for filename in tqdm.tqdm(os.listdir(imgs_path)):
			# Check if the file is a .jpg image
			if filename.endswith(".jpg"):
				# Get the full image path
				image_path = os.path.join(imgs_path, filename)
				final_image, mask = remove_background(image_path)

				# Get the base filename without extension
				base_name = os.path.splitext(filename)[0]
				mask_filename = f"{base_name}_mask.png"
				mask_path = os.path.join(imgs_path, mask_filename)

				# Save the mask
				cv.imwrite(mask_path, mask * 255)

				# Save the final image in a masked folder
				if not os.path.exists(os.path.join(imgs_path, "masked")):
					os.makedirs(os.path.join(imgs_path, "masked"))
				final_image_filename = f"masked/{base_name}.jpg"
				final_image_path = os.path.join(imgs_path, final_image_filename)
				cv.imwrite(final_image_path, final_image)

	if args.score:
		# Load the ground truth and predicted masks
		ground_truths, predicted_masks = load_masks(imgs_path)

		# Calculate the global F1 score
		global_f1, global_precision, global_recall = global_f1_score(predicted_masks, ground_truths)
		print(f"Global F1 Score: {global_f1}")
		print(f"Global Precision: {global_precision}")
		print(f"Global Recall: {global_recall}")


if __name__ == "__main__":
	main()
