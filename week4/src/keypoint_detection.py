import cv2 as cv
import numpy as np
import glob
import os
from skimage.feature import daisy
import matplotlib.pyplot as plt


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


def daisy_descriptor(img):

    # Convert the image from BGR to grayscale, as DAISY operates on single-channel images
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Compute the DAISY descriptors for the grayscale image
    # Parameters:
    # - step: spacing between descriptors
    # - radius: radius of the outermost ring
    # - rings, histograms, orientations: control the DAISY descriptor structure
    # - visualize: if True, returns a visualization of the descriptors
    descs, descs_img = daisy(gray, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)


    cv.imshow('dst',descs_img)
    cv.waitKey(0)

    # Reshape descriptors to a 2D array and convert to float32 for compatibility
    return descs.reshape(-1, descs.shape[2]).astype(np.float32)


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
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good

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


