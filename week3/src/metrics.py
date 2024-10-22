from enum import Enum

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim


class Metrics(Enum):
	"""
	Enum class to define the different comparison methods for
	histograms.
	"""
	MANHATTAN = 10
	LORENTZIAN = 20
	CANBERRA = 30
	SSIM = 40


def compare_histograms(hist1, hist2, method=cv.HISTCMP_CHISQR) -> float:
	"""
	Compares two histograms using a specified method.
	This function compares two histograms and returns a distance or similarity
	score based on the method chosen.
	:param hist1: First histogram to be compared.
	:param hist2: Second histogram to be compared (of the same size and type as hist1).
	:param method: Comparison method chosen for the application
	:return: distance/similarity score based on the chosen method 
	"""

	if method == Metrics.MANHATTAN:
		dist = np.sum((np.abs(hist1 - hist2)))
	elif method == Metrics.LORENTZIAN:
		dist = np.sum(np.log(1+np.abs(hist1-hist2)))
	elif method == Metrics.CANBERRA:
		dist = np.sum((np.abs(hist1 - hist2) / (hist1 + hist2 + 1e-10)))
	elif method == cv.HISTCMP_HELLINGER:
		dist = np.sum(np.multiply(hist1, hist2))
	elif method == cv.HISTCMP_CHISQR_ALT:
		dist = 2*np.sum((pow((hist1-hist2),2))/(hist1+hist2+1e-10))
	elif method == Metrics.SSIM:
		data_range = hist1.max() - hist1.min()  # Assuming hist1 and hist2 have the same range
		dist, _ = ssim(hist1, hist2, data_range=data_range, full=True)
		
	return dist


def precision(gt_mask,pred_mask):
	"""
	Calculate the precision
	:param gt_mask: Ground truth mask
	:param pred_mask: Predicted mask
	:return: Precision
	"""
	TP = np.sum((pred_mask == 255) & (gt_mask == 255))
	FP = np.sum((pred_mask == 255) & (gt_mask == 0))

	precision = TP/ (TP+FP) if (TP + FP) != 0 else 0

	return precision


def recall(gt_mask,pred_mask):
	"""
	Calculate the recall
	:param gt_mask: Ground truth mask
	:param pred_mask: Predicted mask
	:return: Recall
	"""
	TP = np.sum((pred_mask == 255) & (gt_mask == 255))
	FN = np.sum((pred_mask == 0) & (gt_mask == 255))

	recall = TP/ (TP+FN) if (TP + FN) != 0 else 0

	return recall


def f1_measure(gt_mask, pred_mask):
	"""
	Calculate the F1 measure
	:param gt_mask: Ground truth mask
	:param pred_mask: Predicted mask
	:return: The F1 measure
	"""
	p = precision(gt_mask,pred_mask)
	r = recall(gt_mask,pred_mask)
	
	f1_measure = (2*p*r)/(p + r) if (p+r) !=0 else 0

	return f1_measure


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


def example():
#EXAMPLE
	gt_mask = cv.imread('PUT_PATH_TO_GROUND_TRUTH_MASK')
	pred_mask = cv.imread('PUT_PATH_TO_PREDICTED_MASK')

	print('Precision: ', precision(gt_mask,pred_mask))
	print('Recall: ', recall(gt_mask,pred_mask))
	print('F1-measure', f1_measure(gt_mask,pred_mask))