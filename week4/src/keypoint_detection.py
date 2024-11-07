import cv2 as cv
import numpy as np
import glob
import os
from skimage.feature import daisy
import matplotlib.pyplot as plt
from scipy.spatial import distance


def sift(img):

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create(nfeatures=400)
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

def orb_daisy_desc(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    desc1 = daisy_descriptor(img)
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=200)

    # find the keypoints with ORB
    kp_orb = orb.detect(gray, None)

    # Compute descriptors using DAISY
    patch_size = 50
    descs = []
    if len(kp_orb) == 0:
        y = int(gray.shape[0]//2)
        x = int(gray.shape[1]//2)
        patch = gray[max(0, y-patch_size//2):y+patch_size//2, max(0, x-patch_size//2):x+patch_size//2]
        desc, descs_img = daisy(patch, step=25, radius=15, rings=3, histograms=6, orientations=10, visualize=True)
        descs.append(np.reshape(desc, -1))
    else:
        for kp in kp_orb:
            x = int(kp.pt[0])
            y = int(kp.pt[1])

            patch = gray[max(0, y-patch_size//2):y+patch_size//2, max(0, x-patch_size//2):x+patch_size//2]
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                desc, descs_img = daisy(patch, step=25, radius=15, rings=3, histograms=6, orientations=10, visualize=True)
                descs.append(np.reshape(desc, -1))

    descs = np.array(descs)

    return descs


def daisy_descriptor(img):

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    h,w = gray.shape

    S = np.floor(w/15)
    R = 15
    Q = 3
    H = 5

    descs = daisy(gray, step=int(S), radius=R, rings=Q, histograms=H, orientations=H, visualize=False)
    #print(descs.shape)
    # Reshape to have a list of feature vectors
    #descs_reshaped = descs.reshape(-1, descs.shape[-1])
    # descs dimension --> (P,Q,R) 
    #cv.imshow('dst', descs_img)
    #cv.waitKey(0)
    #print(descs_reshaped.shape)

    return descs, descs.shape


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
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1.astype(np.float32),des2.astype(np.float32),k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good

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


    return ratios[good_matches], good_matches, sorted_dists

def test_daisy():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder_path = os.path.join(base_path, "./data/qsd1_w4/images_without_noise/masked/00018_1.jpg")
    folder_path_bbdd = os.path.join(base_path, "./data/BBDD")
    img1 = cv.imread(folder_path)

    descs1 = orb_daisy_desc(img1)


  
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


