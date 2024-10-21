import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
import pywt


def dct_block_histogram(image, total_blocks=16, bins=16):
    """
    Computes the concatenated DCT histograms for an image divided into a specified number of blocks.

    :param image: Input grayscale image. If the input image is in color, it will be converted to grayscale.
    :param total_blocks: Total number of blocks to divide the image into (must be a perfect square).
    :param bins: Number of bins for the histogram.
    :return: 1D array containing the concatenated histograms of all blocks.
    """
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Get the image dimensions (height and width)
    height, width = gray_image.shape[:2]

    # Determine block size
    if total_blocks == 1:
        blocks_per_dim = 1
        block_height = height
        block_width = width
    else:
        blocks_per_dim = int(np.sqrt(total_blocks))
        block_height = height // blocks_per_dim
        block_width = width // blocks_per_dim

    # Loop over each block and compute the DCT histogram
    descriptors = []
    for n in range(blocks_per_dim):
        for m in range(blocks_per_dim):
            # Define the current block using slicing
            block = gray_image[n*block_height:(n+1)*block_height, m*block_width:(m+1)*block_width]

            # Compute the DCT for the block
            dct = cv.dct(np.float32(block))

            # Compute the histogram of the DCT coefficients
            hist, _ = np.histogram(dct.ravel(), bins=bins)
            hist = np.float32(hist)
            eps = 1e-7
            hist /= (hist.sum() + eps)

            # Append the histogram for this block
            descriptors = np.concatenate([descriptors, hist])

    return descriptors


def lbp_block_histogram(image, total_blocks=16, R=1, bins = 16):
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
            
            
            (hist, _) = np.histogram(lbp.ravel(),bins=bins)
            
            
            hist = np.float32(hist)
            eps = 1e-7
            hist /= (hist.sum() + eps)

           

            # Append the histogram for this block
            histograms = np.concatenate([histograms, hist])

    return histograms


def wavelet_histogram(image, wavelet='db1', bins=16, level=1):
    """
    Computes the concatenated Wavelet histograms for the entire image (without dividing into blocks).

    :param image: Input grayscale image. If the input image is in color, it will be converted to grayscale.
    :param wavelet: Type of wavelet to use ('db1', 'haar', etc.).
    :param bins: Number of bins for the histogram.
    :param level: Level of the wavelet decomposition.
    :return: 1D array containing the concatenated histograms of all wavelet subbands.
    """
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply wavelet decomposition (DWT) on the entire image
    coeffs = pywt.wavedec2(gray_image, wavelet, level=level)

    # Extract subbands (approximation and details)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]

    # List to store histograms
    descriptors = []

    # Compute histograms for each subband (approximation and details)
    for coeff in [cA, cH, cV, cD]:
        hist, _ = np.histogram(coeff.ravel(), bins=bins)
        hist = np.float32(hist)
        eps = 1e-7
        hist /= (hist.sum() + eps)  # Normalize the histogram

        # Append the histogram for this subband
        descriptors = np.concatenate([descriptors, hist])

    return descriptors





