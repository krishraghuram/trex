import cv2
import numpy as np

im = cv2.imread("balloons.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(im, None)

im_kp = np.zeros(im.shape, dtype=np.uint8)

###This works
im_kp = cv2.drawKeypoints(im,kp,None)

###This does not work
# cv2.drawKeypoints(im, kp, im_kp)

cv2.namedWindow("Key", cv2.WINDOW_NORMAL)
cv2.imshow("Key", im_kp)
cv2.imwrite("temp.png", im_kp)
cv2.waitKey(0)
