import numpy as np
import matplotlib.pyplot as plt

def likelihood(x,mean,cov):
	'''
	Returns the likelihood of observation x belonging to a gaussian class with given mean and covariance
	'''
	return np.exp(-0.5*np.matmul(np.matmul((x-mean),(np.linalg.inv(cov))),(x-mean).T))/(np.linalg.det(cov)**0.5)

def classifier(x,training_params):
	'''
	Classifies data x into one of 3 gaussian classes by maximising the likelihood
	'''
	mean1,mean2,mean3,cov1,cov2,cov3 = training_params
	L1 = likelihood(x,mean1,cov1)
	L2 = likelihood(x,mean2,cov2)
	L3 = likelihood(x,mean3,cov3)
	
	if L1>=L2 and L1>L3:
		return 1
	elif L2>L1 and L2>=L3:
		return 2
	else:
		return 3


def training(alpha): 
	'''
	Will train a classifier and then return the train and test error. For smooth plots, we will compute the average training and test error
	'''
	av_testing_err = 0
	av_training_err = 0
	n_iter = 100 # Number of iterations to average over for finding the training and test error
	for j in range(n_iter):
		# Generate data
		u1 = np.array([0,0,0])
		u2 = np.array([1,5,-3])
		u3 = np.array([0,0,0])
		sigma1 = np.diag([3,5,2])
		sigma2 = np.array([[1,0,0],[0,4,1],[0,1,6]])
		sigma3 = np.diag([10,10,10])

		N = 20 # Number of training data samples from each class
		x1 = np.random.multivariate_normal(u1,sigma1,N)
		x2 = np.random.multivariate_normal(u2,sigma2,N)
		x3 = np.random.multivariate_normal(u3,sigma3,N)
		x = np.append(x1,np.append(x2,x3,0),0)

		# Estimating means of the distributions
		s_mean1 = np.mean(x1,0)
		s_mean2 = np.mean(x2,0)
		s_mean3 = np.mean(x3,0)

		# Estimating covariances of the distributions
		s_cov1 = np.cov(x1.T)
		s_cov2 = np.cov(x2.T)
		s_cov3 = np.cov(x3.T)

		cov = np.cov(x.T) # Common covariance matrix which will be used to shrink the estimated covariance matrices

		#print("Sample means are: {}, {} and {}".format(s_mean1,s_mean2,s_mean3))
		#print("Sample covariance matrices are \n {} \n {} \n {}".format(s_cov1,s_cov2,s_cov3))

		# Shrinking the estimated covariance matrices
		cov_final1 = ((1-alpha)*N*s_cov1 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)
		cov_final2 = ((1-alpha)*N*s_cov2 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)
		cov_final3 = ((1-alpha)*N*s_cov3 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)

		# The following will used to maximise the likelihood of new unseen data
		training_params = s_mean1,s_mean2,s_mean3,cov_final1,cov_final2,cov_final3

		# Finding training error
		training_err = 0
		for i in range(3*N):
			label_est = classifier(x[i,:],training_params)
			if label_est!=i//N + 1: # First 20 are in class 1, next 20 are in class 2 and so on
				training_err+=1
		training_err = training_err/3/N # Average training error

		# Testing
		N = 50
		x1 = np.random.multivariate_normal(u1,sigma1,N)
		x2 = np.random.multivariate_normal(u2,sigma1,N)
		x3 = np.random.multivariate_normal(u3,sigma1,N)
		x = np.append(x1,np.append(x2,x3,0),0)

		# Finding testing error
		testing_err = 0
		for i in range(3*N):
			label_est = classifier(x[i,:],training_params)
			if label_est!=i//N + 1:
				testing_err+=1
		testing_err = testing_err/3/N

		av_training_err += training_err/n_iter
		av_testing_err += testing_err/n_iter

	return av_training_err,av_testing_err


def main():
	training_err = np.array([]) # Will store the training errors for different alpha
	testing_err = np.array([]) # Will store the test error for different alpha
	n_alpha = 100 # Number of values of alpha in (0,1) for which the code will be run

	for alpha in range(1,n_alpha,1):
		err = training(alpha/n_alpha)
		training_err = np.append(training_err,err[0])
		testing_err = np.append(testing_err,err[1])
	
	# Plotting the variation of error with alpha
	plt.title("Plot of error variation with alpha")
	plt.plot(training_err,'b',label = "Training Error")
	plt.plot(testing_err,'r',label = "Testing Error")
	plt.xlabel(r'Alpha $\rightarrow$')
	plt.ylabel(r'Error $\rightarrow$')
	plt.grid()
	plt.legend()
	#plt.savefig('plots/error')
	plt.show()

if __name__ == "__main__":
	main()
	