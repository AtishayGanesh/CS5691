import numpy as np 
import matplotlib.pyplot as plt

def main():
	"""
	mean = np.random.uniform(1,100)
	sigma = 0.5
	x = np.random.normal(mean,sigma,(100,100))/100
	"""

	c = np.array([np.arange(1,101)]).T/101
	x = np.random.normal(np.repeat(c,100,1),5e-1) # use 0.12 for 0.8
	x = np.random.normal(np.random.uniform(size=(100,100)),0.1,(100,100))

	#for i in range(1,100):
	#	x[:,i] = np.roll(x[:,i-1],1)

	corr = np.corrcoef(x.T)
	#print(np.linalg.matrix_rank(x))
	print(corr) # Put limit on 0 and 1
	u,s,_ = np.linalg.svd(x)
	
	# PART A
	fro_norm = np.linalg.norm(x,ord = 'fro')
	print("Frobenius norm: {}".format(fro_norm))

	# PART B
	large_sing_vals = np.linalg.norm(s[:10])
	print("Fraction of frobenius norm captured by the largest 10 singular values: {}".format(large_sing_vals/fro_norm))

	print("largest",s[0])
	# PART C
	s_square = np.square(s,s)
	summ = np.zeros(10)
	for i in range(10):
		summ[i] = np.sum(s_square[:i+1])**0.5
	plt.plot(summ/fro_norm)
	plt.show()

if __name__ == "__main__":
	main()