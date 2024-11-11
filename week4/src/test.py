import cv2 as cv
import os
from keypoint_detection import *

'''
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, "./data/qsd1_w4/images_without_noise/masked")
folder_path_bbdd = os.path.join(base_path, "./data/BBDD")


# Get all images with .jpg extension
image_path_1 = os.path.join(folder_path, "00001_0.jpg")

image_path_2 = os.path.join(folder_path_bbdd, "bbdd_00097.jpg")

image_path_3 = os.path.join(folder_path_bbdd, "bbdd_00150.jpg")

img_1 = cv.imread(image_path_1)

img_1 = cv.resize(img_1, (256, 256))

cv.imshow("Imagen 1", img_1)
cv.waitKey(0)

kp0, des0 = sift(img_1)

kp0, des0 = orb(img_1)

des0 = orb_daisy_desc(img_1)
'''


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, "./data/qsd1_w4/images_without_noise/masked")
folder_path_bbdd = os.path.join(base_path, "./data/BBDD")

# Get all images with .jpg extension
image_path_1 = os.path.join(folder_path, "00001_0.jpg")

image_path_2 = os.path.join(folder_path_bbdd, "bbdd_00024.jpg")

image_path_3 = os.path.join(folder_path_bbdd, "bbdd_00150.jpg")

img1 = cv.imread(image_path_1)
#img1 = cv.resize(img1, (256, 256))

kp1, des1 = sift(img1)

print(kp1)

img2 = cv.imread(image_path_2)
#img2 = cv.resize(img2, (256, 256))

kp2, des2 = sift(img2)


img3 = cv.imread(image_path_3)
#img3 = cv.resize(img3, (256, 256))

kp3, des3 = sift(img3)


matches = match(des1, des2, 'sift')
print(len(matches))

img12 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img12_rgb = cv.cvtColor(img12, cv.COLOR_BGR2RGB)
plt.imshow(img12_rgb)
plt.show()


matches = match(des1, des3, 'sift')
print(len(matches))

img13 = cv.drawMatches(img1,kp1,img3,kp3,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

matches = match(des3, des1, 'sift')
img31 = cv.drawMatches(img3,kp3,img1,kp1,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img13_rgb = cv.cvtColor(img13, cv.COLOR_BGR2RGB)
plt.imshow(img13_rgb)
plt.show()

img31_rgb = cv.cvtColor(img31, cv.COLOR_BGR2RGB)
plt.imshow(img31_rgb)
plt.show()

