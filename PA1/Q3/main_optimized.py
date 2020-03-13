import numpy as np
import matplotlib.pyplot as plt

def likelihood(x,mean,cov):
	return np.exp(-0.5*np.matmul(np.matmul((x-mean).T,(np.linalg.inv(cov))),(x-mean)))/(np.linalg.det(cov)**0.5)

def classifier(x,training_params):
	mean1,mean2,mean3,cov1,cov2,cov3 = training_params
	L1 = likelihood(x,mean1,cov1)
	L2 = likelihood(x,mean2,cov2)
	L3 = likelihood(x,mean3,cov3)
	print(L1.shape)
	# What if they are equal?
	return np.argmax(np.append(L1,np.append(L2,L3,axis=1),axis=1),axis=1)+1
	if L1>L2 and L1>L3:
		return 1
	elif L2>L1 and L2>L3:
		return 2
	else:
		return 3


def training(alpha): 
	av_testing_err = 0
	av_training_err = 0
	n_iter = 1
	for j in range(n_iter):
		u1 = np.array([0,0,0])
		u2 = np.array([1,5,-3])
		u3 = np.array([0,0,0])
		sigma1 = np.diag([3,5,2])
		sigma2 = np.array([[1,0,0],[0,4,1],[0,1,6]])
		sigma3 = np.diag([10,10,10])

		N = 20
		x1 = np.random.multivariate_normal(u1,sigma1,N)
		x2 = np.random.multivariate_normal(u2,sigma1,N)
		x3 = np.random.multivariate_normal(u3,sigma1,N)
		x = np.append(x1,np.append(x2,x3,0),0)

		mean = np.mean(x)
		s_mean1 = np.mean(x1,0)
		s_mean2 = np.mean(x2,0)
		s_mean3 = np.mean(x3,0)

		s_cov1 = np.cov(x1.T)
		s_cov2 = np.cov(x2.T)
		s_cov3 = np.cov(x3.T)

		x_ed = np.append(x1-s_mean1,np.append(x2-s_mean2,x3-s_mean3,0),0)
		cov = np.cov(x.T) # We will use this to skrink the covariance matrices

		#print("Sample means are: {}, {} and {}".format(s_mean1,s_mean2,s_mean3))

		scale = 3
		cov_final1 = ((1-alpha)*N*s_cov1 + alpha*(scale*N)*cov)/((1-alpha)*N + alpha*scale*N)
		cov_final2 = ((1-alpha)*N*s_cov2 + alpha*(scale*N)*cov)/((1-alpha)*N + alpha*scale*N)
		cov_final3 = ((1-alpha)*N*s_cov3 + alpha*(scale*N)*cov)/((1-alpha)*N + alpha*scale*N)
		
		training_params = s_mean1,s_mean2,s_mean3,cov_final1,cov_final2,cov_final3

		# Finding training error
		training_err = 0
		for i in range(3*N):
			label_est = classifier(x[i,:],training_params)
			if label_est!=i//N + 1:
				training_err+=1
		training_err = training_err*100/3/N

		# Testing
		N = 50
		x1 = np.random.multivariate_normal(u1,sigma1,N)
		x2 = np.random.multivariate_normal(u2,sigma1,N)
		x3 = np.random.multivariate_normal(u3,sigma1,N)
		x = np.append(x1,np.append(x2,x3,0),0)

		# Finding testing error
		testing_err = 0
		label_est = classifier(x,training_params)
		print(label_est)
		for i in range(3*N):
			if label_est!=i//N + 1:
				testing_err+=1
		testing_err = testing_err*100/3/N

		av_training_err += training_err/n_iter
		av_testing_err += testing_err/n_iter
	return av_training_err,av_testing_err


def main():
	training_err = np.array([])
	testing_err = np.array([])
	n_alpha = 30
	for alpha in range(1,n_alpha,1+n_alpha):
		print(alpha)
		err = training(alpha/n_alpha)
		training_err = np.append(training_err,err[0])
		testing_err = np.append(testing_err,err[1])
	plt.plot(training_err,'b')
	plt.plot(testing_err,'r')
	plt.show()

if __name__ == "__main__":
	main()
	