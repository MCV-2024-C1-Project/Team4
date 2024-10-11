import cv2 as cv
import argparse
import numpy as np
import os
import tqdm


def fill_surrounded_pixels(foreground):
	"""
	Fill the surrounded pixels in the foreground
	:param foreground: The foreground mask
	:return: The new mask with surrounded pixels filled
	"""
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
		upper_bound = np.percentile(S_border_pixels, 97)  # 97th percentile
		S_border_pixels = S_border_pixels[(S_border_pixels <= upper_bound)]
	if np.min(V_border_pixels) < 255 * 0.30:
		lower_bound = np.percentile(V_border_pixels, 3)  # 3th percentile
		V_border_pixels = V_border_pixels[(V_border_pixels >= lower_bound)]

	return np.max(S_border_pixels), np.min(V_border_pixels)


def remove_background(image_path):
	"""
	Remove the background from an image
	:param image_path: Path to the image
	:return: The image with the background removed and the mask
	"""
	# Read image
	image = cv.imread(image_path)

	myimage_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	threshold_s, threshold_v = get_s_and_v_masks(myimage_hsv)

	# Take S and remove any value that is less than half
	s = myimage_hsv[:, :, 1]
	s = np.where(s < threshold_s + 1, 0, 1)  # Any value below 127 will be excluded

	# We increase the brightness of the image and then mod by 255
	v = myimage_hsv[:, :, 2]
	v = np.where(v > threshold_v - 1, 0, 1)  # Any value above 255 will be part of our mask

	# Combine our two masks based on S and V into a single "Foreground"
	foreground = np.where((s == 0) & (v == 0), 0, 1).astype(np.uint8)
	foreground[:10, :] = 0;
	foreground[-10:, :] = 0;
	foreground[:, :10] = 0;
	foreground[:, -10:] = 0
	foreground = fill_surrounded_pixels(foreground)

	kernel = np.ones((5, 5), np.uint8)
	opening = cv.morphologyEx(foreground, cv.MORPH_OPEN, kernel)
	kernel = np.ones((25, 25), np.uint8)
	foreground = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

	# Show background mask
	background = np.where(foreground == 0, 255, 0).astype(np.uint8)  # Invert foreground to get background in uint8
	background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)  # Convert background back into BGR space

	img_foreground = cv.bitwise_and(image, image, mask=foreground)  # Apply our foreground map to original image

	# Show the final image
	return img_foreground - background, foreground  # Combine foreground and background


def global_f1_score(masks, ground_truths):
	"""
	Calculate the global F1 score for a dataset of masks and ground truths.

	Args:
	- masks (list of np.array): List of mask arrays.
	- ground_truths (list of np.array): List of corresponding ground truth arrays.

	Returns:
	- global_f1 (float): The global F1 score for the dataset.
	"""
	total_tp, total_fp, total_fn = 0, 0, 0

	for pred_mask, gt_mask in zip(masks, ground_truths):
		# Calculate TP, FP, FN for each mask
		tp = np.sum((pred_mask == 255) & (gt_mask == 255))
		fp = np.sum((pred_mask == 255) & (gt_mask == 0))
		fn = np.sum((pred_mask == 0) & (gt_mask == 255))

		total_tp += tp
		total_fp += fp
		total_fn += fn

	# Calculate global precision and recall
	global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
	global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

	# Calculate global F1 score
	if global_precision + global_recall > 0:
		global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)
	else:
		global_f1 = 0

	return global_f1, global_precision, global_recall


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

	if not args.score:
		# Loop through all files in the directory
		for filename in tqdm.tqdm(os.listdir(args.imgs_path)):
			# Check if the file is a .jpg image
			if filename.endswith(".jpg"):
				# Get the full image path
				image_path = os.path.join(args.imgs_path, filename)
				finalimage, mask = remove_background(image_path)

				# Get the base filename without extension
				base_name = os.path.splitext(filename)[0]
				mask_filename = f"{base_name}_mask.png"
				mask_path = os.path.join(args.imgs_path, mask_filename)

				# Save the mask
				cv.imwrite(mask_path, mask * 255)

	if args.score:
		# Load the ground truth and predicted masks
		ground_truths, predicted_masks = load_masks(args.imgs_path)

		# Calculate the global F1 score
		global_f1, global_precision, global_recall = global_f1_score(predicted_masks, ground_truths)
		print(f"Global F1 Score: {global_f1}")
		print(f"Global Precision: {global_precision}")
		print(f"Global Recall: {global_recall}")


if __name__ == "__main__":
	main()
