import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
import pywt
import matplotlib.pyplot as plt


def dct_block_histogram(image, total_blocks=16, bins=16, N=None):
    """
    Computes the concatenated DCT histograms for an image divided into a specified number of blocks.

    :param image: Input grayscale image. If the input image is in color, it will be converted to grayscale.
    :param total_blocks: Total number of blocks to divide the image into (must be a perfect square).
    :param bins: Number of bins for the histogram. If N is specified, this parameter is ignored.
    :param N: Number of DCT coefficients to keep (zig-zag scanned). If specified, the histogram bins will be N.
    :return: 1D array containing the concatenated histograms of all blocks.
    """
    assert total_blocks > 0, "The number of blocks must be greater than 0."
    assert N is None or N > 0, "The number of DCT coefficients must be greater than 0."

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

            if N is not None:
                # Zig-zag scan the DCT coefficients
                dct = np.concatenate([np.diagonal(dct[::-1, :], k)[::(2*(k % 2)-1)] for k in range(1-dct.shape[0], dct.shape[0])])
                dct = dct[:N]

                # Compute the histogram of the DCT coefficients
                hist, _ = np.histogram(dct.ravel(), bins=N)
            else:
                # Compute the histogram of all DCT coefficients
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


def resize_image(image, size=(256, 256)):
    # Resize the image to a fixed size
    return cv.resize(image, size, interpolation=cv.INTER_AREA)

def wavelet_descriptor(image, wavelet, level):
    # Resize the image to a fixed size
    image = resize_image(image, size=(256, 256))
    
    # Convert the image to grayscale if it's in RGB
    if len(image.shape) == 3:
        image = np.mean(image, -1)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    
    # Normalize the approximation coefficients (coeffs[0]) to scale between -1 and 1
    coeffs[0] /= np.abs(coeffs[0]).max()

    # Normalize the detail coefficients at each level, avoiding division by zero
    for detail_level in range(level):
        normalized_subbands = []
        for i in range(len(coeffs[detail_level + 1])):
            subband = coeffs[detail_level + 1][i]
            max_val = np.abs(subband).max()
            if max_val != 0:  # Ensure no division by zero
                subband = subband / max_val
            normalized_subbands.append(subband)
        coeffs[detail_level + 1] = tuple(normalized_subbands)  # Reassign the normalized subbands
    
    # Convert the list of wavelet coefficients into a 2D array for use as a descriptor
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Flatten the array to use it as a feature vector
    descriptor = arr.flatten()
    
    return descriptor

