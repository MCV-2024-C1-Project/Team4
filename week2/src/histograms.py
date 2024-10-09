import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from compute_similarities import compute_similarities
from compute_descriptors import compute_histogram
from mpl_toolkits.mplot3d import Axes3D
from metrics import Metrics

def compute_histogram3d(img, channels, bins, ranges, normalized: bool = True):
    hist = cv.calcHist([img], channels, None, bins, ranges)
    if normalized:
        cv.normalize(hist, hist)
    return hist

def block_histogram(img,total_blocks,bins,ranges):

    h, w = img.shape[:2]

    #Divide the image into blocks of size

    blocks_per_dim = int(total_blocks / 2)
    block_size_x = w // blocks_per_dim
    block_size_y = h // blocks_per_dim

    histograms = []

    for y in range(blocks_per_dim):
        for x in range(blocks_per_dim):
            block = img[y*block_size_y:(y+1)*block_size_y , x*block_size_x:(x+1)*block_size_x]
            hist = compute_histogram3d(block,[0, 1, 2], [bins, bins, bins], ranges, normalized= True)
            histograms.append(hist)
    return histograms 

# Visualizar los histogramas 3D de cada bloque
def plot_histograms_3d(histograms, blocks_per_dim, bins):
      # Crear la cuadrícula de bins
    bin_edges = np.arange(bins+1)
    x, y, z = np.meshgrid(bin_edges[:-1], bin_edges[:-1], bin_edges[:-1])

    # Ravel las dimensiones para obtener los puntos 3D
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # Crear subplots para cada histograma
    for idx, hist in enumerate(histograms):
        fig = plt.figure(figsize=(8, 8))  # Tamaño de la figura para cada bloque
        ax = fig.add_subplot(111, projection='3d')
        
        hist_values = hist.ravel()
        


        # Crear los puntos del histograma como scatter plot
        scatter = ax.scatter3D(x, y, z, c=hist_values, cmap='viridis', marker='o',s=hist_values * 50)
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        colorbar.set_label('Probability')
        
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        ax.set_title(f'Block {idx+1}')

    plt.tight_layout()
    plt.show()
'''
image = cv.imread('../../data/qsd1_w2/qsd1_w1/00001.jpg')
image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
color_space = "HSV"
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bbdd_path = os.path.join(base_path, "data", "BBDD")
with open(os.path.join(bbdd_path, color_space+'_histograms.pkl'), 'rb') as f:
		bbdd_histograms = pickle.load(f)

block_size = 4  
bins = 32
ranges = [0,180,0,256,0,256]
k_value = 5
histograms = block_histogram(image, block_size, bins,ranges)
similarity_function = Metrics.CANBERRA
print(compute_similarities(histograms, bbdd_histograms, similarity_function, k_value))

'''







