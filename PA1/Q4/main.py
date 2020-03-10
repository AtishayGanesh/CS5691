import numpy as np 

def main():
	mean = np.random.uniform(1,100)
	sigma = 0.5
	x = np.random.normal(mean,sigma,(100,100))/100
	u,s,_ = np.linalg.svd(x)
	
	# PART A
	fro_norm = np.linalg.norm(x,ord = 'fro')
	print("Frobenius norm: {}".format(fro_norm))

	# PART B
	large_sing_vals = np.linalg.norm(s[:10])
	print("Fraction of frobenius norm captured by the largest 10 singular values: {}".format(large_sing_vals/fro_norm))

	# PART C


if __name__ == "__main__":
	main()