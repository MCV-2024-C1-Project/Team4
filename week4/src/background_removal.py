import cv2 as cv
import argparse
import numpy as np
import os
import tqdm

from metrics import global_f1_score
from utils import order_points, plot_mask_with_points


def fill_surrounded_pixels(foreground):
	"""
	Fill the surrounded pixels in the foreground
	:param foreground: The foreground mask
	:return: The new mask with surrounded pixels filled
	"""
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


def get_artworks_points(mask):
	"""
	Get the points of the two largest artworks in the mask
	:param mask: The mask of the image
	:return: The points of the two largest artworks (if they exist)
	"""
	# Step 1: Find contours
	contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# Step 2: Take the two largest contours (assuming the mask has two main objects)
	contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]

	# Step 3: Get area of the two largest contours
	if len(contours) < 2:
		return [contours[0]]

	# Step 4: Return the largest contour if the second largest is less than 90% of the largest otherwise return both
	area1 = cv.contourArea(contours[0])
	area2 = cv.contourArea(contours[1])
	if (area1 - area2) > area1*0.95:
		return [contours[0]]

	return contours


def sort_contours(contours):
	"""
	Sort contours from top to bottom and left to right.
	Only sort left to right if contours are close in vertical alignment.

	:param contours: List of contours
	:return: Sorted list of contours
	"""

	# Calculate the centroid or top-left corner of bounding box for each contour
	def get_key(contour):
		x, y, w, h = cv.boundingRect(contour)
		return (y, x)  # Sort primarily by y (top to bottom), then by x (left to right)

	# Sort contours by y first, then by x, with an additional check for same-row alignment
	sorted_contours = sorted(contours, key=get_key)

	# Ensure contours on the same row are sorted left to right
	final_sorted = []
	current_row = []
	current_y = None

	for contour in sorted_contours:
		x, y, w, h = cv.boundingRect(contour)

		# Start a new row if y difference is significant (not in the same row)
		if current_y is None or abs(y - current_y) > h / 2:  # New row threshold
			if current_row:
				# Sort the current row by x (left to right) before adding to final list
				current_row.sort(key=lambda c: cv.boundingRect(c)[0])
				final_sorted.extend(current_row)
				current_row = []
			current_y = y  # Set the new row reference

		current_row.append(contour)

	# Add the last row after sorting
	if current_row:
		current_row.sort(key=lambda c: cv.boundingRect(c)[0])
		final_sorted.extend(current_row)

	return final_sorted


def get_s_and_v_masks(hsv_image):
	"""
	Get the S and V masks for the image
	:param hsv_image: The HSV image
	:return: The S and V masks
	"""
	# Separate the HSV channels
	H, S, V = cv.split(hsv_image)

	# Extract top 10 rows, bottom 10 rows, left 10 columns, and right 10 columns
	top_S = S[:10, :]
	bottom_S = S[-10:, :]
	left_S = S[:, :10]
	right_S = S[:, -10:]

	top_V = V[:10, :]
	bottom_V = V[-10:, :]
	left_V = V[:, :10]
	right_V = V[:, -10:]

	# Combine each channel's border pixels separately
	S_border_pixels = np.concatenate((top_S.flatten(), bottom_S.flatten(), left_S.flatten(), right_S.flatten()))
	V_border_pixels = np.concatenate((top_V.flatten(), bottom_V.flatten(), left_V.flatten(), right_V.flatten()))

	if np.max(S_border_pixels) > 255 * 0.70:
		upper_bound = np.percentile(S_border_pixels, 80)  # 97th percentile
		S_border_pixels = S_border_pixels[(S_border_pixels <= upper_bound)]
	if np.min(V_border_pixels) < 255 * 0.30:
		lower_bound = np.percentile(V_border_pixels, 20)  # 3th percentile
		V_border_pixels = V_border_pixels[(V_border_pixels >= lower_bound)]

	return np.max(S_border_pixels), np.min(V_border_pixels)


def old_bg_removal(image_hsv):
	threshold_s, threshold_v = get_s_and_v_masks(image_hsv)

	# Take S channel and create a mask so that each pixel with a saturation level below or equal
	# to the threshold is set to 0 (background) and S_levels > th are set to 1 (painting)
	s = image_hsv[:, :, 1]
	s = np.where(s < threshold_s + 1, 0, 1)

	# Take V channel and create a mask so that each pixel with a Value level above or equal
	# to the threshold is set to 0 (background) and V_levels < th are set to 1 (painting)
	v = image_hsv[:, :, 2]
	v = np.where(v > threshold_v - 1, 0, 1)  # Any value above 255 will be part of our mask

	# Combine our two masks based on S and V into a single "Foreground"
	foreground = np.where((s == 0) & (v == 0), 0, 1).astype(np.uint8)
	foreground[:10, :] = 0
	foreground[-10:, :] = 0
	foreground[:, :10] = 0
	foreground[:, -10:] = 0
	foreground = fill_surrounded_pixels(foreground)

	# Opening -> removes foreground objects smaller than the kernel
	kernel = np.ones((5, 5), np.uint8)
	opening = cv.morphologyEx(foreground, cv.MORPH_OPEN, kernel)

	return opening


def apply_square_mask(mask):
	# Image dimensions
	height, width = mask.shape

	# Step 1: Find the first row with more white than black from the top
	top_row = next(i for i in range(height) if np.sum(mask[i, :] == 1) > 0)

	# Step 2: Find the first row with more white than black from the bottom
	bottom_row = next(i for i in range(height - 1, -1, -1) if np.sum(mask[i, :] == 1) > 0)

	# Step 3: Find the first column with more white than black from the left
	left_col = next(j for j in range(width) if np.sum(mask[:, j] == 1) > 0)

	# Step 4: Find the first column with more white than black from the right
	right_col = next(j for j in range(width - 1, -1, -1) if np.sum(mask[:, j] == 1) > 0)

	# Create a new mask to draw the square
	mask = np.zeros_like(mask, dtype=np.uint8)

	# Fill the area inside the square with white
	mask[top_row:bottom_row + 1, left_col:right_col + 1] = 1
	return mask


def remove_background(image_path):
	"""
	Remove the background from an image
	:param image_path: Path to the image
	:return: The image with the background removed and the mask
	"""
	# Step 1: GrabCut to remove the background
	# Read image
	image = cv.imread(image_path)
	image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	#cv.imshow("Original image", image)
	#cv.waitKey(0)

	old_bg_mask = old_bg_removal(image)
	#cv.imshow("Foreground 1", old_bg_mask*255)
	#cv.waitKey(0)

	# Create an empty mask
	mask = np.ones(image.shape[:2], np.uint8)
	mask[old_bg_mask == 0] = cv.GC_BGD  # Surely background
	mask[old_bg_mask > 0] = cv.GC_PR_FGD  # Probable foreground

	# Define the edges of the image as definite background
	mask[:10, :] = cv.GC_BGD  # Top 10 rows
	mask[-10:, :] = cv.GC_BGD  # Bottom 10 rows
	mask[:, :10] = cv.GC_BGD  # Left 10 columns
	mask[:, -10:] = cv.GC_BGD  # Right 10 columns

	# Define background and foreground models
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)

	# Apply GrabCut
	cv.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

	# Create mask where sure and likely foreground are 1
	mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')

	#cv.imshow("GrabCut mask", mask2*255)
	#cv.waitKey(0)

	# Step 2: Morphological operations and filling surrounded pixels
	# Opening + Closing to remove noise
	kernel = np.ones((15, 15), np.uint8)
	mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
	foreground = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)

	foreground = fill_surrounded_pixels(foreground)

	#cv.imshow("Foreground - Step 2", foreground*255)
	#cv.waitKey(0)

	# Get the sorted contours of the artworks
	contours = sort_contours(get_artworks_points(foreground))

	dsts = []
	masks = []
	for contour in contours:
		mask = np.zeros_like(foreground)
		cv.drawContours(mask, [contour], -1, (1, 1, 1), thickness=cv.FILLED)
		mask = apply_square_mask(mask)

		#cv.imshow("Mask", mask * 255)
		#cv.waitKey(0)

		# Get image masked
		dst = image * mask[:, :, np.newaxis]

		#cv.imshow("Image masked", dst)
		#cv.waitKey(0)

		# Remove all rows and cols with black pixels (zeros)
		non_zero_rows = np.any(dst != 0, axis=(1, 2))
		non_zero_cols = np.any(dst != 0, axis=(0, 2))
		dst = dst[non_zero_rows][:, non_zero_cols]

		#cv.imshow("Image masked 2", dst)
		#cv.waitKey(0)

		dst = cv.cvtColor(dst, cv.COLOR_HSV2BGR)
		#cv.imshow("Final image", dst)
		#cv.waitKey(0)

		dsts.append(dst)
		masks.append(mask)

	combined_mask = np.any(masks, axis=0).astype(np.uint8)
	return dsts, combined_mask


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
				final_images, mask = remove_background(image_path)

				# Get the base filename without extension
				base_name = os.path.splitext(filename)[0]
				mask_filename = f"{base_name}_mask.png"
				mask_path = os.path.join(imgs_path, mask_filename)

				# Save the mask
				cv.imwrite(mask_path, mask * 255)

				# Save the final images in a masked folder
				if not os.path.exists(os.path.join(imgs_path, "masked")):
					os.makedirs(os.path.join(imgs_path, "masked"))
				for i, final_image in enumerate(final_images):
					final_image_filename = f"masked/{base_name}_{i}.jpg"
					final_image_path = os.path.join(imgs_path, final_image_filename)
					cv.imwrite(final_image_path, final_image)

	if args.score:
		# Load the ground truth and predicted masks
		ground_truths, predicted_masks = load_masks(imgs_path)

		# Calculate the global F1 score
		global_f1, global_precision, global_recall = global_f1_score(predicted_masks, ground_truths)
		print(f"Global F1 Score: {np.round(global_f1, 2)}")
		print(f"Global Precision: {np.round(global_precision, 2)}")
		print(f"Global Recall: {np.round(global_recall, 2)}")


if __name__ == "__main__":
	main()
