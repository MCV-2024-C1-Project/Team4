import os
import cv2 as cv
import argparse

from utils import plot_hist_task1


def calculate_descriptor(img, image_filename, descriptor_type):
    """
    Calculate the histogram based on the descriptor type
    :param image: image to calculate the histogram
    :param descriptor_type: descriptor type (e.g., hist_lab or hist_hsv)
    """
    if descriptor_type == 'hist_lab':
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        plot_hist_task1(img,image_filename, img_lab, 'Lab')
    elif descriptor_type == 'hist_hsv':
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        plot_hist_task1(img,image_filename, img_hsv, 'HSV')
    else:
        print(f"Descriptor {descriptor_type} not recognized. Use 'hist_lab' or 'hist_hsv'.")


# Main function
def main():
    # Script arguments
    parser = argparse.ArgumentParser(description="Calculate image descriptors")
    parser.add_argument("query_path", help="Path to the query dataset")
    parser.add_argument("image_filename", help="Image filename (e.g., 00001.jpg)")
    parser.add_argument("descriptor_type", help="Descriptor type (e.g., hist_lab or hist_hsv)")
    args = parser.parse_args()

    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory
    q_path = os.path.join(base_path, args.query_path)

    image_path = os.path.join(q_path, args.image_filename)

    # Load the image
    img = cv.imread(image_path)
    
    if img is None:
        print(f"Could not load image {args.image_filename}")
        return

    # Calculate descriptor
    calculate_descriptor(img, args.image_filename, args.descriptor_type)

if __name__ == "__main__":
    main()
