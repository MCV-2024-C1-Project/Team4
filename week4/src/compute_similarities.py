from typing import Any

from metrics import compare_histograms
from skimage import color, data
from skimage.feature import daisy
from scipy.spatial import distance
import cv2 as cv
import numpy as np  
from metrics import Metrics
from keypoint_detection import * 


def compute_similarities_bidirectional(query_descriptors: Any, bbdd_descriptors: Any, des_type: str, k: int = 1) -> tuple[list, list]:
    results = []
    
    for idx, bbdd_desc in enumerate(bbdd_descriptors):
        if bbdd_desc is None:
            num_good_matches = 0
            results.append((idx, num_good_matches))
        else:
            # Get matches from query to BBDD
            matches_query_to_bbdd = match(query_descriptors, bbdd_desc, des_type)
            num_good_matches = len(matches_query_to_bbdd)

            # If there are matches, check for bidirectional matching
            if num_good_matches > 0:
                # Check if matches are reciprocal
                matches_bbdd_to_query = match(bbdd_desc, query_descriptors, des_type)
                if len(matches_bbdd_to_query) > 0:
                    # If there's a valid reciprocal match, count it as a good match
                    num_good_matches = min(num_good_matches, len(matches_bbdd_to_query))
                else:
                    num_good_matches = 0  # No reciprocal match found, set to zero

            results.append((idx, num_good_matches))
    
    # Sort results by the number of good matches in descending order
    # (Higher number of matches indicates a better similarity)
    results.sort(key=lambda x: x[1], reverse=True)

    # Extract only the indices of the sorted results
    results_idx = [result[0] for result in results]

    # Set a threshold value based on descriptor type to help identify unknown images
    threshold = 1.7 if des_type == 'sift' or des_type == 'daisy' else 1.8

    # Check if the top match is significantly better than the second-best match
    # This helps to identify cases where there may be no good match in the dataset
    if len(results) > 1 and results[0][1] < threshold * results[1][1] or results[0][1] == 0:
        # Insert a placeholder (-1) result at the top to indicate an unknown image
        result = (-1, -1)
        results.insert(0, result)
        results_idx.insert(0, -1)

    # Return the top k matches (with their similarity scores and indices)
    return results[:k], results_idx[:k]



def compute_similarities(query_descriptors: Any, bbdd_descriptors: Any, des_type: str, k: int = 1) -> tuple[list, list]:
    """
    Computes the similarities between the query descriptors and the BBDD descriptors using bidirectional matching.
    
    :param query_descriptors: descriptors of the query image.
    :param bbdd_descriptors: list of descriptors for each image in the BBDD.
    :param des_type: type of descriptor (e.g., 'sift', 'orb', 'daisy').
    :param k: number of top results to return.
    :return: top k results and the indices of the associated images in the BBDD.
    """

    # Initialize an empty list to store the results of matches
    results = []
    
    # Loop through each descriptor in the reference (BBDD) descriptors list
    for idx, bbdd_desc in enumerate(bbdd_descriptors):
        # If the current descriptor is None, set the number of good matches to 0
        if bbdd_desc is None:
            num_good_matches = 0
            results.append((idx, num_good_matches))
        else:
            # Compute bidirectional matches between the query and the current reference descriptor
            matches = match(query_descriptors, bbdd_desc, des_type)
            
            # Count the number of good bidirectional matches
            num_good_matches = len(matches)
            
            # Calculate the standard deviation of match distances (helps to gauge match quality)
            #std_dev = compute_std_dev_of_distances(matches) if matches else float('inf')
            
            # Append the index of the reference descriptor, the number of good matches, and the standard deviation
            #results.append((idx, num_good_matches, std_dev))
            results.append((idx, num_good_matches))

    # Sort results by the number of good matches in descending order
    # (Higher number of matches indicates a better similarity)
    results.sort(key=lambda x: x[1], reverse=True)

    # Extract only the indices of the sorted results
    results_idx = [result[0] for result in results]

    # Set a threshold value based on descriptor type to help identify unknown images
    threshold = 1.7 if des_type == 'sift' else 1.8

    # Check if the top match is significantly better than the second-best match
    # This helps to identify cases where there may be no good match in the dataset
    if len(results) > 1 and results[0][1] < threshold * results[1][1]:
        # Insert a placeholder (-1) result at the top to indicate an unknown image
        result = (-1, -1, -1)
        results.insert(0, result)
        results_idx.insert(0, -1)

    # Return the top k matches (with their similarity scores and indices)
    return results[:k], results_idx[:k]


def compute_similarities_daisy(query_descriptors: Any, bbdd_descriptors: Any, k: int = 1) -> tuple[list, list]:
    best_match_index = -1
    min_score = float('inf')
    distances = []
    mins = []
    th = 0.83

    for idx, bbdd_desc in enumerate(bbdd_descriptors):
        if len(bbdd_desc) == 0:
            if len(query_descriptors) == 0:
                min_score = 0
                best_match_index = idx
        else:
            for kp_desc in query_descriptors:
                dist = np.linalg.norm(bbdd_desc-kp_desc, axis=1)
                distances.append(np.sort(dist)[0])
            
            if len(distances) > 1:
                local_scores_vector = np.reshape(distances, -1)
                local_scores_vector = np.sort(local_scores_vector)
                mean0 = np.mean(local_scores_vector[:1])
            else:
                local_scores_vector = distances[0]
                mean0 = local_scores_vector
            
            if mean0 < min_score:
                mins.append(min_score)
                min_score = mean0
                best_match_index = idx
    
    ratio = min_score/np.max([min(mins), 1e-8])
    if ratio > th:
        best_match_index = -1
    
    return min_score, best_match_index



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