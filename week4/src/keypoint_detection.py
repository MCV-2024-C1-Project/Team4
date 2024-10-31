import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
import pywt
import matplotlib.pyplot as plt

def harris_corner_detector():