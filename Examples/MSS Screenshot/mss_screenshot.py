import time
import cv2
import mss
import numpy

with mss.mss() as sct:
	# Part of the screen to capture
	monitor = {'top': 280, 'left': 680, 'width': 620, 'height': 160}
	# monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
	
	i = 0
	while i<1000:
	# while 'Screen capturing':
		last_time = time.time()

		# Get raw pixels from the screen, save it to a Numpy array
		im = numpy.array(sct.grab(monitor))

		#Save the picture
		i = i+1
		print cv2.imwrite('mss/'+str(i)+'.jpg', im)
		time.sleep(0.05)

		# Display the picture
		# cv2.imshow('OpenCV/Numpy normal', im)
		# Display the picture in grayscale
		# cv2.imshow('OpenCV/Numpy grayscale', cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY))
		# print('fps: {0}'.format(1 / (time.time()-last_time)))

		# Press "q" to quit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break