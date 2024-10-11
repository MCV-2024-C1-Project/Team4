import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from compute_similarities import compute_similarities
from compute_descriptors import compute_histogram
from mpl_toolkits.mplot3d import Axes3D
from metrics import Metrics
from average_precision import *

def compute_histogram3d(img, channels, bins, ranges, normalized: bool = True):
    """
    Computes the 3D histogram of an image.
    
    :param img: Image for which the 3D histogram has to be computed.
    :param channels: Channels of the color space for which the histogram is computed.
    :param bins: Number of bins for each channel.
    :param ranges: Range of values for each channel.
    :param normalized: If True, normalize the histogram. Default is True.
    :return: Tuple containing the computed histogram and a 1D array representation of it.
    """
    hist = cv.calcHist([img], channels, None, bins, ranges)
    if normalized:
        cv.normalize(hist, hist)
    return hist, hist.flatten()

def block_histogram(img,total_blocks,bins,ranges):
    """
    Computes the block histogram of an image by dividing it into smaller blocks.
    
    :param img: Image to be divided into blocks.
    :param total_blocks: Total number of blocks to divide the image into.
    :param bins: Number of bins for each channel in the histogram.
    :param ranges: Range of values for each channel.
    :return: 1D array containing the concatenated histograms of all blocks.
    """

    # Get the image dimensions (height and width respectively)
    h, w = img.shape[:2]

    # Determine the size of the blocks

    if total_blocks == 1:
        blocks_per_dim = 1
        block_size_m = w 
        block_size_n = h 
    else:
        blocks_per_dim = int(np.sqrt(total_blocks))
        block_size_m = w // blocks_per_dim
        block_size_n = h // blocks_per_dim

    histograms = []
    # Iterate over each block to compute its histogram
    for n in range(blocks_per_dim):
        for m in range(blocks_per_dim):
            # Define the current block using slicing
            block = img[n*block_size_n:(n+1)*block_size_n , m*block_size_m:(m+1)*block_size_m]
            # Compute the 3D histogram for the current block
            hist, hist_vect = compute_histogram3d(block,[0, 1, 2], [bins, bins, bins], ranges, normalized= True)
            # Concatenate the 3D histograms computed for each block of the image
            histograms = np.concatenate([histograms, hist_vect])
            # Uncomment below to visualize the 3D histogram for each block
            #print("1st block 3D Histogram")
            #plot_histogram_3d(hist, bins)
    return histograms

def spatial_pyramid_histogram(img, num_levels, bins, ranges):
    """
    Computes the spatial pyramid histogram of an image.
    
    :param img: Image for which the spatial pyramid histogram is computed.
    :param num_levels: Number of levels in the pyramid.
    :param bins: Number of bins for each channel in the histogram.
    :param ranges: Range of values for each channel.
    :return: 1D array containing the concatenated histograms of all levels.
    """
    histograms = []

    # Get the image dimensions
    h, w = img.shape[:2]
    
    for level in range(num_levels):
        # Calculate the number of blocks per dimension at this level
        blocks_per_dim = 2 ** level   # base ** exponent -> 2^(level)
        
        block_size_h = h // blocks_per_dim
        block_size_w = w // blocks_per_dim
        
        for n in range(blocks_per_dim):
            for m in range(blocks_per_dim):
                # Define the current block using slicing
                block = img[n * block_size_h : (n + 1) * block_size_h,
                             m * block_size_w : (m + 1) * block_size_w]
                
                # Compute the 3D histogram for the current block
                hist, hist_vect = compute_histogram3d(block, [0, 1, 2], [bins, bins, bins], ranges)
                
                # Concatenate the histograms
                histograms = np.concatenate([histograms, hist_vect])
    
    return histograms


def plot_histogram_3d(histogram, bins):
    """
    Plots a 3D histogram.

    :param histogram: A 3D histogram as a NumPy array. This array should contain the frequency of pixel values.
    :param bins: The number of bins for each dimension of the histogram.
    :return: None. This function displays a 3D plot of the histogram.
    """
    
    # Create the bin edges for each dimension
    bin_edges = np.arange(bins+1) # Creates an array of edges for the bins
    x, y, z = np.meshgrid(bin_edges[:-1], bin_edges[:-1], bin_edges[:-1])  # Create a meshgrid for 3D coordinates

    # Flatten the meshgrid coordinates to get 3D points
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') # Add a 3D subplot to the figure

    hist_values = histogram.ravel() # Flatten the histogram array to get the frequency values


    # Create a scatter plot for the histogram points
    scatter = ax.scatter3D(x, y, z, c=hist_values, cmap='viridis', marker='o',s=hist_values * 50)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10) # Add a color bar to indicate probability
    colorbar.set_label('Probability')

    # Set axis labels    
    ax.set_xlabel('H')
    ax.set_ylabel('s')
    ax.set_zlabel('V')
    plt.tight_layout()
    plt.show()


#EXEMPLE BINS = 64, DISTANCE = CAMBERRA , ESPAI DE COLOR = HSV
def exemple():

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bbdd_path = os.path.join(base_path, "data", "BBDD")
    q_path = os.path.join(base_path, "data","qsd1_w1")
    q_path = os.path.join(base_path, "data","qsd1_w1")
    color_space = "HSV"
    with open(os.path.join(bbdd_path, color_space+'_histograms.pkl'), 'rb') as f:
            bbdd_histograms = pickle.load(f)
    with open(q_path + "/gt_corresps.pkl", 'rb') as f:
                y = pickle.load(f)

    files = [f for f in os.listdir(q_path) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    total_blocks = 4
    histograms = []
    for filename in files:

            # Read image (by default the color space of the loaded image is BGR) 
        img_bgr = cv.imread(os.path.join(q_path, filename))

            # Change color space (only 2 options are possible)
        if color_space == "Lab":
            img = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
            bins_channel1 = 64
            ranges = [0,256,0,256,0,256]
        elif color_space == "HSV":
            img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
            bins_channel1 = 64
            ranges = [0,180,0,256,0,256]

        hist = block_histogram(img,total_blocks,bins_channel1,ranges)

        index = int(filename.split('_')[-1].split('.')[0])

        if len(histograms) <= index:
            histograms.extend([None] * (index + 1 - len(histograms)))
        histograms[index] = hist

        
    with open(os.path.join(q_path, color_space + '_histograms.pkl'), 'wb') as f:
        pickle.dump(histograms, f)


    with open(os.path.join(q_path, color_space+'_histograms.pkl'), 'rb') as f:
            query_histograms = pickle.load(f)

    k_value = 5
    similarity_measure = "Canberra"
    similarity_function = Metrics.CANBERRA

    res_m = []
        
    for query_img_h in query_histograms:
        res_m.append(compute_similarities(query_img_h, bbdd_histograms, similarity_function, k_value)[1])
        
        # If we are not in testing mode

            # Save the top K indices of the museum images with the best similarity for each query image to a pickle file
        with open(os.path.join(q_path, color_space +'_'+ similarity_measure + '_' + str(k_value) + '_results.pkl'), 'wb') as f:
            pickle.dump(res_m, f)

            # Evaluate the results using mAP@K if we are not in testing mode	
        print(f"mAP@{k_value} for {color_space}: {mapk(y, res_m, k_value)}")


def example2():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q_path = os.path.join(base_path, "data","qsd1_w1")
    color_space = "HSV"


    files = [f for f in os.listdir(q_path) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    total_blocks = 4
    histograms = []

    for filename in files[:2]:

        # Read image (by default the color space of the loaded image is BGR) 
        img_bgr = cv.imread(os.path.join(q_path, filename))

        img = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        plt.imshow(img)
        img[:,:,2] = cv.equalizeHist(img[:,:,2])
        plt.imshow(img)
        bins_channel1 = 8
        ranges = [0,180,0,256,0,256]

        hist = block_histogram(img,total_blocks,bins_channel1,ranges)
        dist = compute_similarities(hist, [hist, hist], similarity_measure=cv.HISTCMP_HELLINGER, k=1)

        index = int(filename.split('_')[-1].split('.')[0])

        if len(histograms) <= index:
            histograms.extend([None] * (index + 1 - len(histograms)))
        histograms[index] = hist

