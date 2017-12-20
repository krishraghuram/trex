import cv2
import numpy as np

#################################################################################
#Read Images
#################################################################################
im1 = cv2.imread("trex1.png", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("trex2.png", cv2.IMREAD_GRAYSCALE)

#################################################################################
#SIFT 
#################################################################################
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)
im_kp1 = np.zeros(im1.shape, dtype=np.uint8)
im_kp2 = np.zeros(im2.shape, dtype=np.uint8)
im_kp1 = cv2.drawKeypoints(im1,kp1,None)
im_kp2 = cv2.drawKeypoints(im2,kp2,None)

#################################################################################
#BF Matching
#################################################################################
### Without Ratio Test
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(des1,des2)
### With Ratio Test
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1,des2, k=2)
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)


matches = good_matches
matches = sorted(matches, key = lambda x:x.distance)
matches = matches[0:100]
im_matches = cv2.drawMatches(im_kp1, kp1, im_kp2, kp2, matches, None, flags=2)

#################################################################################
#Display Results
#################################################################################
cv2.namedWindow("Key1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Key2", cv2.WINDOW_NORMAL)
cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
while True:
	cv2.imshow("Key1", im_kp1)
	cv2.imshow("Key2", im_kp2)
	cv2.imshow("Matches", im_matches)
	c = cv2.waitKey(30)
	if c==27:
		break
