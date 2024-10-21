import cv2
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import shutil
import pickle

# Task 1: Noise filtering on images

# Argument parser
parser = argparse.ArgumentParser(description="Filter noise from images.")
parser.add_argument("query_path", help="Path to the query dataset")
parser.add_argument("--filter", type=str, choices=['median', 'nlm', 'gaussian'], 
                    help="Type of filter to apply: 'median', 'nlm', or 'gaussian'", default="median")
args = parser.parse_args()

# Path to the folder containing the images
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, args.query_path)

# Get all images with .jpg extension
image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

# Store filtered images for later display
filtered_images = {}

# Create output directory if it doesn't exist
output_dir = os.path.join(folder_path, "images_without_noise")
os.makedirs(output_dir, exist_ok=True)

# List to keep track of images with noise
images_with_noise = []

# Function to apply the selected filter
def apply_filter(img, filter_type):
    if filter_type == 'median':
        return cv2.medianBlur(img, 5)  # Median filter (kernel size 5)
    elif filter_type == 'nlm':
        return cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 3, 15)  # Non-local Means filter
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(img, (7, 7), 0)  # Gaussian blur (kernel size 7x7)

# Apply filters and compute quality metrics for each image
for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
    # Read the image
    img = cv2.imread(image_path)
    
    # Apply the chosen filter
    filtered_img = apply_filter(img, args.filter)

    # Convert images to grayscale for SSIM calculation
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM (Structural Similarity Index)
    ssim_value, _ = ssim(gray_original, gray_filtered, full=True)
    # SSIM of 1 indicates that the images are identical; 0 indicates no structural correlation.

    if ssim_value > 0.56:
        # Save original image
        shutil.copy(image_path, os.path.join(output_dir, os.path.basename(image_path)))
    else:
        # If SSIM is below threshold, save the median filtered image
        images_with_noise.append(os.path.basename(image_path))
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), filtered_img)


# Print the list of images with noise
print("Images with noise:")
for img_name in images_with_noise:
    print(img_name)

'''
with open(folder_path + "/augmentations.pkl", 'rb') as f:
			y = pickle.load(f) 
                  
print(y)
'''


