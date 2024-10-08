import os

from compute_descriptors import compute_descriptors

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
	# Compute descriptors for the BBDD images (offline)
	imgs_path = os.path.join(base_path, "data", "BBDD")

	compute_descriptors(imgs_path, color_space="Lab")
	compute_descriptors(imgs_path, color_space="HSV")


if __name__ == "__main__":
	main()
