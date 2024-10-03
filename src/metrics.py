import cv2 as cv


def compare_histograms(hist1, hist2, method=cv.HISTCMP_CHISQR):
	"""
	Compares two histograms using a specified method.

	This function compares two histograms and returns a distance or similarity
	score based on the method chosen.

	Parameters
	----------
	hist1: array-like. First compared histogram.
	hist2: array-like. Second compared histogram of the same size and type as hist1.
	method: Comparison method. The default is 'cv.HISTCMP_CHISQR', which uses the Chi-Square method.

	Returns
	-------
	score: float. Output distance/similarity.
	"""
	return cv.compareHist(hist1, hist2, method)
