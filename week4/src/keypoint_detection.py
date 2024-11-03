import cv2 as cv
import numpy as np
import glob
import os
from skimage.feature import daisy
import matplotlib.pyplot as plt
from scipy.spatial import distance


def sift(img):

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=400)
    
    # find the keypoints and compute the descriptors with SIFT
    kp, des = sift.detectAndCompute(gray,None)

    #img=cv.drawKeypoints(gray,kp,img)

    #cv.imshow('dst',img)
    #cv.waitKey(0)

    return kp, des


def orb(img):

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(gray,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)

    #img=cv.drawKeypoints(gray,kp,img)

    #cv.imshow('dst',img)
    #cv.waitKey(0)

    return kp, des

    import cv2 as cv
import numpy as np


def daisy_descriptor(img):

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    h,w = gray.shape

    S = np.floor(w/20)
    R = 15
    Q = 3
    H = 8

    descs, descs_img = daisy(gray, step=int(S), radius=R, rings=Q, histograms=Q+1, orientations=H, normalization='daisy' ,visualize=True)
    #print(descs.shape)
    # Reshape to have a list of feature vectors
    descs_reshaped = descs.reshape(-1, descs.shape[-1])
    # descs dimension --> (P,Q,R) 
    #cv.imshow('dst', descs_img)
    #cv.waitKey(0)
    #print(descs_reshaped.shape)

    return descs_reshaped


def compute_std_dev_of_distances(matches):
    """
    Compute the standard deviation of distances for a list of cv2.DMatch objects.

    :param matches: List of cv2.DMatch objects
    :return: Standard deviation of distances
    """
    distances = [match[0].distance for match in matches]
    return np.std(distances)


def match(des1, des2, des_type):

    if des_type == 'sift':

        #bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        #matches = bf.match(des1,des2)
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1,des2,k=2)

    elif des_type == 'orb':

        #bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        #matches = bf.match(des1,des2)
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(des1,des2,k=2)

    elif des_type == 'daisy':

        #bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        #matches = bf.match(des1,des2)
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1,des2,k=2)


    good = []
    for match_pair in matches:
        # Check if we have two matches for this descriptor
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:  # Apply the ratio test
                good.append(m)  # Append the best match only

    return good

def bidirectional_match(query_descriptors, bbdd_descriptors, des_type):
    # Create a BFMatcher object based on the descriptor type
    if des_type == 'sift' or des_type == 'orb':
        bf = cv.BFMatcher(cv.NORM_L2 if des_type == 'sift' else cv.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError("Descriptor type not supported for bidirectional matching")

    # Perform forward matching (query to BBDD)
    matches_q_to_bbdd = bf.knnMatch(query_descriptors, bbdd_descriptors, k=1)

    # Perform reverse matching (BBDD to query)
    matches_bbdd_to_q = bf.knnMatch(bbdd_descriptors, query_descriptors, k=1)

    # Filter matches that are mutual in both directions (bidirectional matches)
    bidirectional_matches = []
    for m in matches_q_to_bbdd:
        if len(m) == 1:
            match = m[0]
            # Check if this match is in the reverse match set
            reverse_match = matches_bbdd_to_q[match.trainIdx]
            if len(reverse_match) == 1 and reverse_match[0].trainIdx == match.queryIdx:
                bidirectional_matches.append(match)

    return bidirectional_matches


def daisy_match(descs1, descs2):

    # Compute pairwise distances between descriptors
    dists12 = distance.cdist(descs1, descs2, 'euclidean')

    # Find the closest matches
    min_dist_indices12 = np.argmin(dists12, axis=1)

    # Compute the two smallest distances for each descriptor in desc1
    sorted_dists = np.sort(dists12, axis=1)
    ratios = sorted_dists[:, 0] / (sorted_dists[:, 1] + 1e-8)  # Adding small value to avoid division by zero

    # Threshold for Lowe's ratio test
    ratio_threshold = 0.6
    good_matches = ratios < ratio_threshold


    return ratios[good_matches]

def test_daisy():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder_path = os.path.join(base_path, "./data/qsd1_w4")
    folder_path_bbdd = os.path.join(base_path, "./data/BBDD")
    image_path_1 = os.path.join(folder_path, "00001.jpg")
    image_path_2 = os.path.join(folder_path_bbdd, "bbdd_00150.jpg")
    image_path_3 = os.path.join(folder_path_bbdd, "bbdd_00003.jpg")
    img1 = cv.imread(image_path_1)
    img2 = cv.imread(image_path_2)
    img3 = cv.imread(image_path_3)

    descs1 = daisy_descriptor(img1).astype(np.float32)
    descs2 = daisy_descriptor(img2).astype(np.float32)
    descs3 = daisy_descriptor(img3).astype(np.float32)

    match(descs1, descs2, 'daisy')
    match(descs1, descs3, 'daisy')

  
#test_daisy()
'''
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, "./data/qsd1_w4")
folder_path_bbdd = os.path.join(base_path, "./data/BBDD")

# Get all images with .jpg extension
image_path_1 = os.path.join(folder_path, "00001.jpg")

image_path_2 = os.path.join(folder_path_bbdd, "bbdd_00097.jpg")

image_path_3 = os.path.join(folder_path_bbdd, "bbdd_00150.jpg")

img1 = cv.imread(image_path_1)

kp1, des1 = sift(img1)

print(kp1)

img2 = cv.imread(image_path_2)
kp2, des2 = sift(img2)


img3 = cv.imread(image_path_3)
kp3, des3 = sift(img3)


matches = match(des1, des2, 'sift')
print(len(matches))

img12 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img12),plt.show()



matches = match(des1, des3, 'sift')
print(len(matches))

#img13 = cv.drawMatches(img1,kp1,img3,kp3,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img13 = cv.drawMatchesKnn(img1,kp1,img3,kp3,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img13),plt.show()
'''


