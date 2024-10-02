import cv2 as cv


def compare_histograms(hist1, hist2, method=cv.HISTCMP_CHISQR):
	return cv.compareHist(hist1, hist2, method)
