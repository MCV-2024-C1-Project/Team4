import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def block_histogram(img,total_blocks,bins):

    h, w = img.shape[:2]

    #Divide the image into blocks of size

    blocks_per_dim = int(np.sqrt(total_blocks))
    block_size_x = w // blocks_per_dim
    block_size_y = h // blocks_per_dim

    histograms = []

    for y in range(blocks_per_dim):
        for x in range(blocks_per_dim):
            block = img[y*block_size_y:(y+1)*block_size_y , x*block_size_x:(x+1)*block_size_x]
            histograms.append(cv.calcHist([block], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]))
    return histograms 



