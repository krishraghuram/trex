import cv2
import numpy as np
import mss
import time
import pyautogui
import constants
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
Returns a list data, of length n
Each item is a dict, consisting of : time, action, image, kp, des, matches
time - the time at which this event occurred
action - user action performed at this moment
image - screenshot taken at this moment
kp, des - sift keypoints and sift descriptor for this image
matches - sift matches between this image and the prev image
'''
def play_and_record():
	sct = mss.mss()

	### Play and Record
	data = []
	while True:
		###Sleep for some time
		time.sleep(constants.loop_sleep_time)
		### Temporary data variable
		temp = {}
		### Randomly choose an action, and perform the action. 
		u = np.random.uniform(0,1)
		if u > 1-constants.space_chance : 
			pyautogui.typewrite(['space'], interval=0.01)
			temp['action'] = 'space'
		### Get image
		im = np.array(sct.grab(constants.monitor))
		im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
		temp['image'] = im
		### Get current time
		t = time.time()
		temp['time'] = t

		### Append temp to data
		data.append(temp)

		### Only keep recent items in list
		if len(data) > constants.list_length :
			data.pop(0)
			break

	### SIFT and BF Matching
	for i in range(len(data)):
		### Get image from data
		im = data[i].get('image')

		### Get SIFT Features
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute(im, None)
		data[i]['kp'] = kp
		data[i]['des'] = des
		if i > 1 :	
			kp_prev = data[i-1].get('kp')
			des_prev = data[i-1].get('des')

			### Brute Force Matching with Ratio Test
			if des is not None:
				bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
				matches = bf.knnMatch(des_prev,des, k=2)
				good_matches = []
				for m,n in matches:
					if m.distance < constants.match_reject_thresh * n.distance:
						good_matches.append(m)
				matches = good_matches
				data[i]['matches'] = matches

	return data



if __name__=='__main__':
	time.sleep(2)
	
	data = play_and_record()

	# ### TEST CODE - Show SIFT Matches
	# for i in range(len(data)-1):
	# 	t1 = data[i].get('time')
	# 	t2 = data[i+1].get('time')
	# 	im1 = data[i].get('image')
	# 	im2 = data[i+1].get('image')
	# 	kp1 = data[i].get('kp')
	# 	kp2 = data[i+1].get('kp')
	# 	matches = data[i+1].get('matches')
	# 	im_kp1 = np.zeros(im1.shape, dtype=np.uint8)
	# 	im_kp2 = np.zeros(im2.shape, dtype=np.uint8)
	# 	im_kp1 = cv2.drawKeypoints(im1,kp1,None)
	# 	im_kp2 = cv2.drawKeypoints(im2,kp2,None)
	# 	im_matches = cv2.drawMatches(im_kp1, kp1, im_kp2, kp2, matches, None, flags=2)
	# 	cv2.putText(im_matches, str(t2-t1), (100,300), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0))
	# 	print t2-t1
	# 	cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
	# 	while True:
	# 		cv2.imshow("Matches", im_matches)
	# 		k = cv2.waitKey(30)
	# 		if k == 32: #32 is space
	# 			break
	# 		elif k==-1:
	# 			pass
	# 		else: 
	# 			print k

	### TEST CODE - Time Variance Analysis
	temp = []
	for i in range(len(data)-1):
		t1 = data[i].get('time')
		t2 = data[i+1].get('time')
		temp.append(t2-t1)
	print "Loop Sleep Time : ", constants.loop_sleep_time
	print "TEMP"
	print temp
	print "Mean : ", np.mean(np.array(temp))
	print "Var : ", np.var(np.array(temp)/constants.loop_sleep_time)

	### Perform clustering on (x1,y1,x2,y2) over all image pairs.
	for i in range(1,len(data)):
		#Get keypoints and matches from data
		kp1 = data[i-1].get('kp')
		kp2 = data[i].get('kp')
		matches = data[i].get('matches')

		# #Perform Clustering
		# cluster_data = [ np.array((j.queryIdx,j.trainIdx)) for j in matches ]
		# cluster_data = [ np.array((kp1[j[0]].pt,kp2[j[1]].pt)) for j in cluster_data ]
		# cluster_data = [ np.array((j[0][0],j[0][1],j[1][0],j[1][1])) for j in cluster_data ]
		# cluster_data = np.array(cluster_data)
		# clustering_output = KMeans(n_clusters=constants.n_clusters, n_init=5, max_iter=100).fit(cluster_data)
		# data[i]['clustering_output'] = clustering_output

		# ### TEST CODE - Show SIFT Keypoints and Cluster Centers
		# cluster_image = data[i].get('image')
		# im_kp1 = np.zeros(cluster_image.shape, dtype=np.uint8)
		# im_kp1 = cv2.drawKeypoints(cluster_image,kp2,None)
		# for i in clustering_output.cluster_centers_ : 
		# 	temp = (int(i[2]),int(i[3]))
		# 	cv2.circle(cluster_image, temp, 5, 50, -1)
		# cv2.namedWindow("Cluster Centers", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)
		# cv2.imshow("Cluster Centers", cluster_image)
		# cv2.imshow("SIFT", im_kp1)
		# cv2.waitKey(0)




