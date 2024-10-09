import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from compute_descriptors import compute_histogram


def compute_histogram3d(img, channels, bins, ranges, normalized: bool = True):
    hist = cv.calcHist([img], channels, None, bins, ranges)
    if normalized:
        cv.normalize(hist, hist)
    return hist.flatten()

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
            hist = compute_histogram3d(block,[0, 1, 2], [bins, bins, bins], ranges, normalized=True)
            histograms.append(hist)
    return histograms 


image = cv.imread('../../data/qsd1_w2/qsd1_w1/00001.jpg')


block_size = 4  
bins = 32  

ranges = [0,256,0,256,0,256]

# Obtener la lista de histogramas 3D
histograms = block_histogram(image, block_size, bins,ranges)

print(len(histograms))




