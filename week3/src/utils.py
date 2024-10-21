import cv2 as cv
import numpy as np


def plot_mask_with_points(image, points):
	"""
	Display the mask with points
	:param image: The image to display
	:param points: The points to display
	"""
	# Check that the image is in BGR format
	img = image.copy()
	if len(img.shape) == 2:
		img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

	# Draw points
	for point in points:
		cv.circle(img, tuple(point), 5, (0, 255, 0), -1)

	cv.imshow('Mask with Points', img)
	cv.waitKey(0)


def order_points(pts):
	"""
	Order points in clockwise direction: top-left, top-right, bottom-right, bottom-left
	:param pts: The points to order
	:return: The ordered points
	"""
	# Function to order points in clockwise direction: top-left, top-right, bottom-right, bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	diff = np.diff(pts, axis=1)

	rect[0] = pts[np.argmin(s)]  # top-left
	rect[2] = pts[np.argmax(s)]  # bottom-right
	rect[1] = pts[np.argmin(diff)]  # top-right
	rect[3] = pts[np.argmax(diff)]  # bottom-left

	return rect
