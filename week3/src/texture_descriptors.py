import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os 
import pickle

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





def process_images_in_directory(directory_path):
    # List to hold all histograms
    histograms = []
    files = [f for f in os.listdir(directory_path) if f.endswith(".jpg")]
    
    # Sort filenames based on the number after the last underscore
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort based on the number at the end
    
    # Process each image in the sorted list
    for filename in files:
            file_path = os.path.join(directory_path, filename)
            # Read the image
            image = cv2.imread(file_path,cv2.COLOR_BGR2GRAY)
            if image is not None:
                # Calculate LBP histogram for the image
                blocks = (4,4)
                hist = lbp_block_histogram(image,blocks)
                index = int(filename.split('_')[-1].split('.')[0])
                if len(histograms) <= index:
                    histograms.extend([None] * (index + 1 - len(histograms)))
                    histograms[index] = hist
    with open(os.path.join(directory_path, '_LBP_'+str(blocks)+'_blocks_'+ '.pkl'), 'wb') as f:
        pickle.dump(histograms, f)


# Directory path
directory_path = r'C:/Users/34634/Downloads/C1/Team4/week3/data/qsd1_w3/qsd1_w3'

# Process all images in the directory
histograms = process_images_in_directory(directory_path)



