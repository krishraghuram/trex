import cv2
from trex import play_and_record
import time
import numpy as np

time.sleep(1)

data = play_and_record()

kp1 = data[0].get('kp')
im1 = data[0].get('image')
matches = data[1].get('matches')


# print len(kp1)
# print kp1[0].pt
# print kp1[0].pt[0]

# im_kp1 = np.zeros(im1.shape, dtype=np.uint8)
# im_kp1 = cv2.drawKeypoints(im1,kp1,None)
# cv2.namedWindow("Temp", cv2.WINDOW_NORMAL)
# cv2.imshow("Temp", im_kp1)
# cv2.waitKey(0)

