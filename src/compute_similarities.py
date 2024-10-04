from metrics import compare_histograms


def compute_similarities(query_hist, bbdd_histograms, method, measure: str="distance", k: int = 1):
	"""
	Computes the similarities between the query histogram and the BBDD histograms
	:param query_hist: query histogram
	:param bbdd_histograms: list of BBDD histograms
	:param method: method to compute the similarity
	:param k: number of results to return. Default is 1
	:return: ...
	"""
	results = []
	for idx, bbdd_hist in enumerate(bbdd_histograms):

		distance = compare_histograms(query_hist, bbdd_hist, method)
		results.append((idx, distance))

	if measure == "similarity": results.sort(key=lambda x: x[1], reverse=True)
	else: results.sort(key=lambda x: x[1])

	results_idx = [result[0] for result in results]
	
	return results[:k], results_idx[:k]
