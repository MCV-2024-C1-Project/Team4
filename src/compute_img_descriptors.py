import os
from enum import Enum
from typing import Any

import cv2 as cv
import argparse

from utils import plot_hist_task1


class DescriptorType(Enum):
    HIST_LAB = 'hist_lab'
    HIST_HSV = 'hist_hsv'


def calculate_descriptor(img: Any, image_filename: str, descriptor_type: str) -> None:
    """
    Calculate the histogram based on the descriptor type
    :param img: image to calculate the histogram
    :param image_filename: filename of the image
    :param descriptor_type: descriptor type according to the selected color space (e.g., hist_lab or hist_hsv)
    """
    try:
        # Convert the string descriptor to the DescriptorType enum
        descriptor_enum = DescriptorType(descriptor_type)
    except ValueError:
        print(f"Descriptor {descriptor_type} not recognized. Use 'hist_lab' or 'hist_hsv'.")
        return

    if descriptor_enum == DescriptorType.HIST_LAB:
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        plot_hist_task1(img,image_filename, img_lab, 'Lab')
    elif descriptor_enum == DescriptorType.HIST_HSV:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        plot_hist_task1(img,image_filename, img_hsv, 'HSV')


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
    assert img is not None, f"Image {args.image_filename} not found in {args.query_path}"

    # Calculate descriptor and plot histograms for image img
    calculate_descriptor(img, args.image_filename, args.descriptor_type)

if __name__ == "__main__":
    main()
