import cv2
import numpy as np

im_name = "balloons.jpg"
im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

for sigma_x in [1,3,5,10,15,20,30]:

	gauss_ksize = (21,21)
	# sigma_x = 5
	im2 = cv2.GaussianBlur(im, gauss_ksize, sigma_x)

	lap_ksize = 1
	ddepth = cv2.CV_16S
	im3 = cv2.Laplacian(im2, ddepth, lap_ksize)

	im4 = im3.copy()
	im4 = im4 - im4.min()
	im4 = im4 * 255.0 / im4.max()
	im4 = im4.astype(np.uint8)

	ret,temp1 = cv2.threshold(im4, 127+20, 255, cv2.THRESH_BINARY)
	ret,temp2 = cv2.threshold(im4, 127-20, 255, cv2.THRESH_BINARY_INV)
	im5 = cv2.bitwise_or(temp1, temp2)


	### TEST CODE
	# temp = set()
	# for i in range(im3.shape[0]):
	# 	for j in range(im3.shape[1]):
	# 		temp.add(im3[i,j])
	# sorted(list(temp))

	# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
	# cv2.imshow("Original", im)
	cv2.namedWindow("GaussianBlur", cv2.WINDOW_NORMAL)
	cv2.imshow("GaussianBlur", im2)
	cv2.namedWindow("LoG", cv2.WINDOW_NORMAL)
	cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
	cv2.imshow("LoG", im4)
	cv2.imshow("Thresh", im5)

	cv2.waitKey(0)


