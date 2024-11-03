from typing import Any

from metrics import compare_histograms
import cv2 as cv
from metrics import Metrics
from keypoint_detection import * 
import numpy as np
from skimage import color, data
from skimage.feature import daisy
from scipy.spatial import distance


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
        if bbdd_desc is None:
            num_good_matches = 0  # Si bbdd_desc es None, asignar 0
            results.append((idx, num_good_matches))
        else:
            matches = match(query_descriptors, bbdd_desc, des_type)
            num_good_matches = len(matches)  # Contar las coincidencias
            std_dev = compute_std_dev_of_distances(matches)  # Standard deviation of distances
            results.append((idx, num_good_matches, std_dev))

    # Sort results by the number of good matches in descending order (more matches is better)
    results.sort(key=lambda x: x[1], reverse=True)

    # Get the indices of the results
    results_idx = [result[0] for result in results]

    threshold = 2 if des_type == 'sift' else 1.8

    if len(results) > 1 and results[0][1] < threshold * results[1][1]:
        result = (-1, -1, -1)
        results.insert(0, result)
        results_idx.insert(0, -1)

    # Return the k best matches
    return results[:k], results_idx[:k]


def compute_similarities_daisy(query_descriptors: Any, bbdd_descriptors: Any, k: int = 1) -> tuple[list, list]:
    """
    Computes the similarities between the query descriptors and the BBDD descriptors.
    
    :param query_descriptors: descriptors of the query image.
    :param bbdd_descriptors: list of descriptors for each image in the BBDD.
    :param des_type: type of descriptor (e.g., 'sift', 'orb', 'daisy').
    :param k: number of top results to return.
    :return: top k results and the indices of the associated images in the BBDD.
    """

    best_match_index = -1
    min_score = float('inf')

    for idx, bbdd_desc in enumerate(bbdd_descriptors):
        radius = 5
        local_scores = compute_local_scores(query_descriptors, bbdd_desc, radius)
        total_score = np.sum(local_scores)

        if total_score < min_score:
            min_score = total_score
            best_match_index = idx



    # Return the k best matches
    return min_score, best_match_index


def euclidean_distance(desc1, desc2):
    # Compute Euclidean distance between two descriptors
    return np.linalg.norm(desc1 - desc2)

def neigh_distance(i, j, query, bbdd, radius):
    # Base case: if delta is very small, return a large number to indicate no valid neighbor
    if radius <= 1:
        return float('inf')
   
    # Get the descriptor at point i in the query image
    desc_i = query[i]

    # Find neighbors of j at distance delta in the template
    neighbors = get_neighbors(j, radius, bbdd.shape[:2])
    distances = []

    # Calculate distances to neighbors
    for k in neighbors:
        desc_k = bbdd[k]
        distances.append(euclidean_distance(desc_i, desc_k))
   
    # Find the minimum distance in this neighborhood
    min_dist = min(distances)
    kmin = neighbors[np.argmin(distances)]  # Get the neighbor with minimum distance

    # Recursively compute the neighborhood distance with a smaller radius
    return min(min_dist, neigh_distance(i, kmin, query, bbdd, radius / 2))

def get_neighbors(j, radius, shape):
    # Function to get neighbors around point j within a radius delta
    neighbors = []
    y, x = j
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            if dy**2 + dx**2 <= radius**2:  # Ensure within radius
                ny, nx = y + dy, x + dx
                if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                    neighbors.append((ny, nx))
    return neighbors

def compute_local_scores(query, bbdd, radius):
    out_result = np.zeros(query.shape[:2])
    for i in range(query.shape[0]):
        for j in range(query.shape[1]):
            desc_i = query[i, j]
            desc_j = bbdd[i, j]
           
            # Compute local descriptor distance
            ld = euclidean_distance(desc_i, desc_j)

            # Compute neighborhood distance
            nd = neigh_distance((i, j), (i, j), query, bbdd, radius)

            # Store the minimum of local and neighborhood distances
            out_result[i, j] = min(ld, nd)

    return out_result

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