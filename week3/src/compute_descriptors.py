import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def lbp_block_histogram(image, num_blocks=(4, 4)):
    """
    Compute concatenated LBP histograms for an image divided into specified number of blocks.
    
    :param image: Input grayscale image
    :param num_blocks: Number of blocks in (height_blocks, width_blocks)
    :return: Concatenated histogram of LBP values for the entire image
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    height, width = gray_image.shape
    lbp_histograms = []
    
    # Define the LBP parameters
    P = 8  # Number of circularly symmetric neighbor set points (podeu ajustar això)
    R = 1  # Radius (podeu ajustar això)

    # Calculate block size based on the number of blocks requested
    block_height = height // num_blocks[0]
    block_width = width // num_blocks[1]

    # Loop through the image in defined blocks
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            # Define block boundaries
            i_start = i * block_height
            j_start = j * block_width
            i_end = (i + 1) * block_height if (i + 1) * block_height <= height else height
            j_end = (j + 1) * block_width if (j + 1) * block_width <= width else width
            
            # Extract the block from the image
            block = gray_image[i_start:i_end, j_start:j_end]
            
            # Calculate the LBP for the block
            lbp_matrix = local_binary_pattern(block, P, R, method="uniform")
            
            # Calculate the histogram for the block
            (hist, _) = np.histogram(lbp_matrix.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            hist = hist.astype("float")
            hist /= hist.sum()  # Normalize the histogram
            
            lbp_histograms.append(hist)
    
    # Concatenate all block histograms into a single histogram
    final_histogram = np.concatenate(lbp_histograms)

    return final_histogram

def plot_histogram(histogram):
    """
    Plot the histogram.
    
    :param histogram: Histogram to plot
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram, width=0.5, color='blue', alpha=0.7)
    plt.title('Concatenated LBP Histogram')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(histogram)), labels=range(len(histogram)), rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show() 



# Example usage
image_path = 'C:/Users/34634/Downloads/C1\Team4/week3/data/qsd1_w3/qsd1_w3/00001.jpg'  # Actualitza amb el camí de la teva imatge
image = cv2.imread(image_path)

# Compute the concatenated LBP histogram for a specified number of blocks
num_blocks = (2,2)  # 4 verticals, 4 horitzontals
lbp_histogram = lbp_block_histogram(image, num_blocks)
plot_histogram(lbp_histogram)

# Display the length of the final histogram
print("Length of concatenated LBP histogram:", len(lbp_histogram))


