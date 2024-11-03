import cv2
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Filter noise from images.")
parser.add_argument("query_path", help="Path to the query dataset")
args = parser.parse_args()

# Path to the folder containing the images
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, args.query_path)

# Get all images with .jpg extension
image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

# Create output directory if it doesn't exist
output_dir = os.path.join(folder_path, "images_resized")
os.makedirs(output_dir, exist_ok=True)

# Define the maximum number of pixels allowed (250,000 in this case)
max_pixels = 250000

for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if img is None:
        print(f"Error loading image {image_path}")
        continue

    # While the total number of pixels is greater than max_pixels, resize the image
    while img.shape[0] * img.shape[1] > max_pixels:
        # Calculate new dimensions (halve each dimension, rounding if necessary)
        new_width = img.shape[1] // 2
        new_height = img.shape[0] // 2
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Construct output path and save the resized image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

print("Image resizing complete!")

