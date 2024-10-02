import os

from compute_descriptors import compute_descriptors

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory

bbdd = os.path.join(base_path, "data", "BBDD")

compute_descriptors(bbdd)