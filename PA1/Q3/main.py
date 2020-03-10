import numpy as np

def likelihood(x,mean,cov):
	return exp(-0.5*(x-mean).T*(np.inv(cov))*(x-mean))/np.linalg.det(cov)

def classifier(x,mean1,mean2,mean3,cov1,cov2,cov3):
	L1 = likelihood(x,mean1,cov1)
	L2 = likelihood(x,mean2,cov2)
	L3 = likelihood(x,mean3,cov3)
	
	# What if they are equal?
	if L1>L2 and L1>L3:
		return 1
	elif L2>L1 and L2>L3:
		return 2
	else:
		return 3


def training(alpha): 
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

	mean = np.sum(x,0)/N/3
	s_mean1 = np.sum(x1,0)/N
	s_mean2 = np.sum(x2,0)/N
	s_mean3 = np.sum(x3,0)/N

	s_cov1 = np.matmul((x1-s_mean1).T,x1-s_mean1)/N
	s_cov2 = np.matmul((x2-s_mean2).T,x2-s_mean2)/N
	s_cov3 = np.matmul((x3-s_mean3).T,x3-s_mean3)/N

	cov = np.matmul((x-mean).T,x-mean)/N/3
	x1 = x1/(cov[0,0]**0.5)
	x2 = x2/(cov[1,1]**0.5)
	x3 = x3/(cov[2,2]**0.5)
	s_mean1 = s_mean1/(cov[0,0]**0.5)
	s_mean2 = s_mean2/(cov[1,1]**0.5)
	s_mean3 = s_mean3/(cov[2,2]**0.5)
	x = np.append(x1,np.append(x2,x3,0),0)
	mean = np.sum(x,0)/N/3

	s_cov1 = np.matmul((x1-s_mean1).T,x1-s_mean1)/N
	s_cov2 = np.matmul((x2-s_mean2).T,x2-s_mean2)/N
	s_cov3 = np.matmul((x3-s_mean3).T,x3-s_mean3)/N
	cov = np.matmul((x-mean).T,x-mean)/N/3
	print(s_cov1)

	cov_final1 = ((1-alpha)*N*s_cov1 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)
	cov_final2 = ((1-alpha)*N*s_cov2 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)
	cov_final3 = ((1-alpha)*N*s_cov3 + alpha*(3*N)*cov)/((1-alpha)*N + alpha*3*N)
	
	return s_mean1,s_mean2,s_mean3,cov_final1,cov_final2,cov_final3


if __name__ == "__main__":
	alpha = 0.1
	training_params = training(alpha)