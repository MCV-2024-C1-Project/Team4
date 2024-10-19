
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

# Task 1: Noise filtering on images

# Path to the folder containing the images
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, "./data/qsd2_w3")

# Get all images with .jpg extension
image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

# Store filtered images for later display
filtered_images = {}

# Create output directory if it doesn't exist
output_dir = os.path.join(folder_path, "images_without_noise")
os.makedirs(output_dir, exist_ok=True)

# List to keep track of images with noise
images_with_noise = []

# Apply filters and compute quality metrics for each image
for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
    # Read the image
    img = cv2.imread(image_path)
    
    # Apply different filters
    median_filtered = cv2.medianBlur(img, 5)  # Median filter (kernel size 5)
    nlm_filtered = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)  # Non-local Means filter
    gaussian_filtered = cv2.GaussianBlur(img, (7, 7), 0)  # Gaussian blur (kernel size 7x7)

    #print(f"Image: {os.path.basename(image_path)}")
    
    # Dictionary to hold each filter's name and result
    filters = {
        "Median Filter": median_filtered,
        "Non-local Means": nlm_filtered,
        "Gaussian Blur": gaussian_filtered
    }
    
    # Store results for each image
    filtered_images[os.path.basename(image_path)] = filters
    
    # Loop through the filters and calculate the metrics
    for filter_name, filtered_img in filters.items():
        # Compute MSE (Mean Squared Error)
        mse_value = mean_squared_error(img, filtered_img)
        # A low MSE indicates that the filtered image is similar to the original image.
        
        # Compute PSNR (Peak Signal-to-Noise Ratio)
        psnr_value = peak_signal_noise_ratio(img, filtered_img)
        # A high PSNR indicates that the filtered image is very similar to the original image.
        
        # Convert images to grayscale for SSIM calculation
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM (Structural Similarity Index)
        ssim_value, _ = ssim(gray_original, gray_filtered, full=True)
        # SSIM of 1 indicates that the images are identical; 0 indicates no structural correlation.

        # Print results for each filter
        #print(f"{filter_name} -> MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.2f}")

    
    # Check the SSIM for the median filtered image
    median_ssim = ssim(gray_original, cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY))

    if median_ssim > 0.53:
        # Save original image
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img)
    else:
        # If SSIM is below threshold, save the median filtered image
        images_with_noise.append(os.path.basename(image_path))
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), median_filtered)


# Print the list of images with noise
print("Images with noise:")
for img_name in images_with_noise:
    print(img_name)


"""
# Choose the image to display
# Example: If you want to show the first filtered image in the list [0]
image_to_show = list(filtered_images.keys())[0]  # Change the index to select another image
filters_to_show = filtered_images[image_to_show]

# Display only the filtered images of the selected image
cv2.imshow("Original Image", cv2.imread(os.path.join(folder_path, image_to_show)))
for filter_name, filtered_img in filters_to_show.items():
    cv2.imshow(filter_name, filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""



