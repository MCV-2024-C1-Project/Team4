
import cv2
import os
import argparse
import glob
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

# Task 1: Noise filtering on images

# Argument parser
parser = argparse.ArgumentParser(description="Filter noise from images.")
parser.add_argument("query_path", help="Path to the query dataset")
args = parser.parse_args()

# Path to the folder containing the images
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, args.query_path)
folder_path_images_without_noise = os.path.join(folder_path, "images_without_noise")
folder_path_non_augmented = os.path.join(folder_path, "non_augmented")

# Get all images with .jpg extension
filtered_images = glob.glob(os.path.join(folder_path_images_without_noise, "*.jpg"))
ground_truth = glob.glob(os.path.join(folder_path_non_augmented, "*.jpg"))

# Create a set of image names for comparison
filtered_image_names = {os.path.basename(img): img for img in filtered_images}
ground_truth_names = {os.path.basename(img): img for img in ground_truth}

# List to store SSIM results
ssim_results = []

# Loop through the images with names in the filtered set
for img_name, filtered_img_path in tqdm(filtered_image_names.items(), desc="Calculating SSIM", unit="image"):
    # Check if the ground truth image exists
    if img_name in ground_truth_names:
        ground_truth_img_path = ground_truth_names[img_name]
        
        # Read images
        filtered_img = cv2.imread(filtered_img_path)
        ground_truth_img = cv2.imread(ground_truth_img_path)
        
        # Convert images to grayscale for SSIM calculation
        gray_filtered = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        gray_ground_truth = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_value, _ = ssim(gray_ground_truth, gray_filtered, full=True)
        
        # Store results
        ssim_results.append({
            "Image": img_name,
            "SSIM": ssim_value
        })

# Print SSIM results
print("\nSSIM Results:")
for result in ssim_results:
    print(f"{result['Image']} -> SSIM: {result['SSIM']:.4f}")

