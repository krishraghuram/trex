import cv2
import numpy as np
import mss
import time
import constants
import pyautogui

'''
Returns a list data, of length n
Each item is a dict, consisting of : time, action, image, kp, des, matches
time - the time at which this event occurred
action - user action performed at this moment
image - screenshot taken at this moment
kp, des - sift keypoints and sift descriptor for this image
matches - sift matches between this image and the prev image
'''
def find_player():
	sct = mss.mss()

	data = []
	while True:
		###Sleep for some time
		time.sleep(constants.loop_sleep_time)
		### Temporary data variable
		temp = {}
		### Get time
		t = time.time()
		temp['time'] = t
		### Get action
		u = np.random.uniform(0,1)
		if u > 1-constants.space_chance : 
			pyautogui.typewrite(['space'], interval=0.01)
			temp['action'] = 'space'
		# Get image
		im = np.array(sct.grab(constants.monitor))
		im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
		temp['image'] = im

		### SIFT
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(im, None)
		temp['kp'] = kp
		temp['des'] = des
		if len(data) > 1 :	
			kp_prev = data[-1].get('kp')
			des_prev = data[-1].get('des')

			### BF Matching with Ratio Test
			if des is not None:
				bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
				matches = bf.knnMatch(des_prev,des, k=2)
				good_matches = []
				for m,n in matches:
					if m.distance < constants.match_reject_thresh * n.distance:
						good_matches.append(m)
				matches = good_matches
				temp['matches'] = matches

		### Append temp to data
		data.append(temp)

		### Only keep recent items in list
		if len(data) > constants.list_length :
			data.pop(0)
			break

	return data




if __name__=='__main__':
	time.sleep(2)
	
	data = find_player()


	### TEST CODE
	for i in range(len(data)-1):
		im1 = data[i].get('image')
		im2 = data[i+1].get('image')
		kp1 = data[i].get('kp')
		kp2 = data[i+1].get('kp')
		matches = data[i+1].get('matches')
		im_kp1 = np.zeros(im1.shape, dtype=np.uint8)
		im_kp2 = np.zeros(im2.shape, dtype=np.uint8)
		im_kp1 = cv2.drawKeypoints(im1,kp1,None)
		im_kp2 = cv2.drawKeypoints(im2,kp2,None)
		im_matches = cv2.drawMatches(im_kp1, kp1, im_kp2, kp2, matches, None, flags=2)
		cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
		cv2.imshow("Matches", im_matches)
		cv2.waitKey(100)











