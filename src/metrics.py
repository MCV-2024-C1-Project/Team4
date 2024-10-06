import cv2 as cv
import numpy as np


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

	if method == 10:
		dist =  np.sum((np.abs(hist1 - hist2)))
	elif method == 20:
		dist = np.sum(np.log(1+np.abs(hist1-hist2)))
	elif method == 30:
		dist = np.sum((np.abs(hist1 - hist2) / (hist1 + hist2 + 1e-10)))
	else:
		dist = cv.compareHist(hist1, hist2, method)
	return dist
