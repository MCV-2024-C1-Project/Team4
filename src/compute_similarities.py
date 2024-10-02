import cv2 as cv
import pickle
import numpy as np

from average_precision import mapk
from metrics import compare_histograms


def compute_similarities(query_hist, bbdd_histograms, method, k=1):
    results = []
    for idx, bbdd_hist in enumerate(bbdd_histograms):
        distance = compare_histograms(query_hist, bbdd_hist, method)
        results.append((idx, distance))
    results.sort(key=lambda x: x[1])
    print(results[:k])


