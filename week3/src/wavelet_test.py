import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Get the path of the folder containing the museum dataset (BBDD)
# This assumes your images are stored in the "BBDD" and "qsd1_w3" folders.
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bbdd_path = os.path.join(base_path, "data", "BBDD")
q_path = os.path.join(base_path, "data", "qsd1_w3")

# Set default figure size for plots
plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})  # Update the font size for plots

# Load the image from the query folder
A = imread(os.path.join(q_path, "00000.jpg"))

# Convert the image to grayscale by averaging across the color channels (RGB to grayscale)
B = np.mean(A, -1)

# Perform wavelet decomposition with n levels
n = 3  # Number of decomposition levels
w = 'db1'  # Type of wavelet to use (Daubechies wavelet with 1 vanishing moment)

# Decompose the grayscale image into wavelet coefficients
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# Normalize the approximation coefficients (coeffs[0]) to scale between -1 and 1
coeffs[0] /= np.abs(coeffs[0]).max()

# Normalize the detail coefficients at each level
for detail_level in range(n):
    # Normalize each subband (horizontal, vertical, diagonal) of the detail coefficients
    coeffs[detail_level + 1] = [d / np.abs(d).max() for d in coeffs[detail_level + 1]]

# Convert the list of wavelet coefficients into a 2D array for visualization
arr, coeff_slices = pywt.coeffs_to_array(coeffs)

# Plot the array of coefficients as an image with grayscale color mapping
# The vmin and vmax are set to control the range of values to display for better contrast
plt.imshow(arr, cmap='gray_r', vmin=-0.25, vmax=0.75)

# Display the figure
plt.show()

