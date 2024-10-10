from enum import Enum

import cv2 as cv
import numpy as np


class Metrics(Enum):
	"""
	Enum class to define the different comparison methods for
	histograms.
	"""
	MANHATTAN = 10
	LORENTZIAN = 20
	CANBERRA = 30


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

def example():
#EXAMPLE
	gt_mask = cv.imread('PUT_PATH_TO_GROUND_TRUTH_MASK')
	pred_mask = cv.imread('PUT_PATH_TO_PREDICTED_MASK')

	print('Precision: ', precision(gt_mask,pred_mask))
	print('Recall: ', recall(gt_mask,pred_mask))
	print('F1-measure', f1_measure(gt_mask,pred_mask))