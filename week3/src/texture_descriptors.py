import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os 
import pickle

def lbp_block_histogram(image, total_blocks=16, R=1):
    """
    Computes the concatenated LBP histograms for an image divided into a specified number of blocks.
    
    :param image: Input grayscale image
    :param total_blocks: Total number of blocks to divide the image into (must be a perfect square).
    :param P: Number of circularly symmetric neighbor set points for LBP.
    :param R: Radius of the circle for LBP.
    :return: 1D array containing the concatenated histograms of all blocks.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Get the image dimensions (height and width)
    height, width = gray_image.shape[:2]

    P = 8*R

    # Determine block size
    if total_blocks == 1:
        blocks_per_dim = 1
        block_height = height
        block_width = width
    else:
        blocks_per_dim = int(np.sqrt(total_blocks))  # Assume perfect square number of blocks
        block_height = height // blocks_per_dim
        block_width = width // blocks_per_dim

    histograms = []

    # Loop over each block and compute the LBP histogram
    for n in range(blocks_per_dim):
        for m in range(blocks_per_dim):
            # Define the current block using slicing
            block = gray_image[n*block_height:(n+1)*block_height, m*block_width:(m+1)*block_width]
            
            # Compute the LBP for the block
            lbp = local_binary_pattern(block, P, R, method="default")
            
            # Compute the histogram using cv.calcHist
            (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 2**P + 1),range=(0, 2**P))
            
            
            hist = np.float32(hist)
            eps = 1e-7
            hist /= (hist.sum() + eps)

           

            # Append the histogram for this block
            histograms = np.concatenate([histograms, hist])

    return histograms










