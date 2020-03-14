import numpy as np 
import matplotlib.pyplot as plt

def main():

	# Need to generate highly correlated columns. 

	# For the highly correlated case
	c = np.array([np.arange(1,101)]).T/101 # Mean of columns
	x = np.random.normal(np.repeat(c,100,1),0.12) # Added gaussian noise to the columns to reduce correlation

	# Use the following x for the uncorrelated case
	# x = np.random.normal(np.random.uniform(-0.1,1.1,(100,100)),0.3,(100,100))

	# Ensuring all values are between 0 and 1
	x[np.where(x>1)] = 1
	x[np.where(x<0)] = 0

	corr = np.corrcoef(x.T)
	print("Rank of the matrix is {}".format(np.linalg.matrix_rank(x)))
	print(corr)

	s = np.sort(np.linalg.eig(np.matmul(x,x.T))[0])[::-1]**0.5 # Finding singular values of x
	
	# PART A
	fro_norm = np.linalg.norm(x,ord = 'fro')
	print("Frobenius norm: {}".format(fro_norm))

	# PART B
	large_sing_vals = np.linalg.norm(s[:10])
	print("Fraction of frobenius norm captured by the largest 10 singular values: {}".format(large_sing_vals/fro_norm))
	print("largest",s[0])

	# PART C
	av_contrib = 0
	N_iter = 100
	for i in range(N_iter):
		s_permuted = np.random.permutation(s) # Picking 10 random singular vectors
		contrib_to_frob_norm = np.linalg.norm(s_permuted[:10])
		av_contrib += (contrib_to_frob_norm/fro_norm)/N_iter
	print("Fraction of frobenius norm captured by the random 10 singular values: {}".format(av_contrib))

	# Part D
	s_square = np.square(s)
	summ = np.zeros(100)
	for i in range(100):
		summ[i] = np.sum(s_square[:i+1])**0.5

	plt.title("Contribution to the frobenius norm vs Number of singular vectors")
	plt.plot(summ/fro_norm)
	plt.grid()
	plt.ylabel(r'Contribution to frobenius norm $\rightarrow$')
	plt.xlabel(r'Number of singular vectors $\rightarrow$')
	plt.savefig("Fro_norm_contrib_uncorr")
	plt.show()

if __name__ == "__main__":
	main()