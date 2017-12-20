
from scipy.stats import multivariate_normal
import numpy as np

for c in [1,3,5,10,15,20,30]:
	# x = np.array([2,2], dtype=np.int)
	temp = [ [2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10] ]
	for x in [np.array(i, dtype=np.int) for i in temp]:
		m = np.array([0,0], dtype=np.int)
		print c,multivariate_normal.pdf(x, mean=m, cov=c)
	print "\n"