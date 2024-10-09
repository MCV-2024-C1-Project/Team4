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
	else:
		dist = cv.compareHist(hist1, hist2, method)
	return dist

def precision(gt_mask,pred_mask):
	gt_mask = np.array(gt_mask)
	pred_mask = np.array(pred_mask)

	TP = np.sum((pred_mask == 255) & (gt_mask == 255))
	FP = np.sum((pred_mask == 255) & (gt_mask == 0))

	precision = TP/ (TP+FP) if (TP + FP) != 0 else 0

	return precision

def recall(gt_mask,pred_mask):
	gt_mask = np.array(gt_mask)
	pred_mask = np.array(pred_mask)

	TP = np.sum((pred_mask == 255) & (gt_mask == 255))
	FN = np.sum((pred_mask == 0) & (gt_mask == 255))

	recall = TP/ (TP+FN) if (TP + FN) != 0 else 0

	return recall


def f1_measure(gt_mask, pred_mask):
	p = precision(gt_mask,pred_mask)
	r = recall(gt_mask,pred_mask)
	
	f1_measure = (2*p*r)/(p + r) if (p+r) !=0 else 0

	return f1_measure

def example():
#EXAMPLE
	gt_mask = cv.imread('C:/Users/34634/Downloads/C1/Team4/week2/data/qsd2_w2/qsd2_w1/00000.png')
	pred_mask = cv.imread('C:/Users/34634/Downloads/C1/Team4/week2/data/qsd2_w2/qsd2_w1/mask_00000.png')

	print('Precision: ', precision(gt_mask,pred_mask))
	print('Recall: ', recall(gt_mask,pred_mask))
	print('F1-measure', f1_measure(gt_mask,pred_mask))