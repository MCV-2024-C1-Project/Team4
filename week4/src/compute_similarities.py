from typing import Any

from metrics import compare_histograms
import cv2 as cv
from metrics import Metrics
from keypoint_detection import * 

def compute_similarities(query_descriptors: Any, bbdd_descriptors: Any, des_type: str, k: int = 1) -> tuple[list, list]:
    """
    Computes the similarities between the query descriptors and the BBDD descriptors.
    
    :param query_descriptors: descriptors of the query image.
    :param bbdd_descriptors: list of descriptors for each image in the BBDD.
    :param des_type: type of descriptor (e.g., 'sift', 'orb', 'daisy').
    :param k: number of top results to return.
    :return: top k results and the indices of the associated images in the BBDD.
    """

    results = []
    for idx, bbdd_desc in enumerate(bbdd_descriptors):
        # Use the match function to find the number of matches between query and BBDD descriptors
        matches = match(query_descriptors, bbdd_desc, des_type)
        num_good_matches = len(matches)
        results.append((idx, num_good_matches))  # Save index and number of good matches
        
    # Sort results by the number of good matches in descending order (more matches is better)
    results.sort(key=lambda x: x[1], reverse=True)

    # Get the indices of the results
    results_idx = [result[0] for result in results]

    # Return the k best matches
    return results[:k], results_idx[:k]





'''
class MeasureType:
    DISTANCE = "distance"
    SIMILARITY = "similarity"


def compute_similarities(query_hist: Any, bbdd_histograms: Any, similarity_measure: Any, k: int = 1) -> tuple[list, list]:
    """
    Computes the similarities between a query histogram and the BBDD histograms
    :param query_hist: query histogram.
    :param bbdd_histograms: list of BBDD histograms.
    :param similarity_measure: measure to compute the similarity .
    :param k: number of results to return. 
    :return: top k results and the indices of the associated images in the BBDD
    """

    measure_type = MeasureType.DISTANCE

    # Check if the measure is a similarity measure
    if similarity_measure == cv.HISTCMP_CORREL or similarity_measure == cv.HISTCMP_INTERSECT or similarity_measure == cv.HISTCMP_HELLINGER or similarity_measure == Metrics.SSIM:
        measure_type = MeasureType.SIMILARITY

    results = []
    for idx, bbdd_hist in enumerate(bbdd_histograms):
        # Compute the distance/similarity between the query histogram and the BBDD histogram
        distance = compare_histograms(query_hist, bbdd_hist, similarity_measure)
        results.append((idx, distance))
        # save a tuple conatining the index identifying the image of the BBDD with which the 
        # query image is compared and the distance/similarity score obtained
        

    # Sort the results depending on whether it's a distance or similarity measure
    if measure_type == MeasureType.SIMILARITY:
        results.sort(key=lambda x: x[1], reverse=True)  # Similarity: higher values are better
    else:
        results.sort(key=lambda x: x[1])                # Distance: lower values are better

    # Get the indices of the results
    results_idx = [result[0] for result in results]

    # Return the k best matches
    return results[:k], results_idx[:k]
'''