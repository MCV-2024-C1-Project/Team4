import cv2 as cv
import numpy as np
import glob
import os
from skimage.feature import daisy
import matplotlib.pyplot as plt


def sift(img):

    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
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
    
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    descs, descs_img = daisy(gray, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)

    cv.imshow('dst',descs_img)
    cv.waitKey(0) 

    return descs.reshape(-1, descs.shape[2]).astype(np.float32)

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
    for m,n in matches:
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


