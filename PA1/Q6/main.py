import numpy as np 
import matplotlib.pyplot as plt

def func(x):
	return np.exp(np.tanh(2*np.pi*x))

def hyper_parameter_tuning(data_train,data_validation):
	# data = data[data[:,0].argsort()]
	N = 70 # Size of dataset for validation
	data_indices = np.arange(0,70,70//N)[:N]
	data = data_train[data_indices]
	alpha = 0
	degrees = [1,3,6,9]
	A = np.ones((N,10))
	A_validation = np.ones((10,10))
	y_reg = np.zeros((N,4))
	validation_err = np.zeros(4)
	for i in range(0,10):
		A[:,i] = data[:,0]**i
		A_validation[:,i] = data_validation[:,0]**i
	for d in range(4):
		A_d = A[:,:degrees[d]+1]
		w = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_d.T,A_d) + alpha*np.eye(degrees[d]+1)),A_d.T),data[:,1])
		y_reg[:,d] = np.matmul(A_d,w)
		validation_err[d] = np.linalg.norm(data_validation[:,1] - np.matmul(A_validation[:,:degrees[d]+1],w))

	plt.plot(data[:,0],data[:,1],'bo')
	plt.plot(data[:,0],y_reg[:,0],'r')
	plt.plot(data[:,0],y_reg[:,1],'g')
	plt.plot(data[:,0],y_reg[:,2],'y')
	plt.plot(data[:,0],y_reg[:,3],'k')
	plt.show()

	plt.plot(validation_err)
	plt.show()

	return degrees[np.argmin(validation_err)]

def main():
	N = 100
	x = np.arange(101)[1:]/101
	y = func(x) + np.random.normal(0,0.2**0.5,N)
	data = np.append(np.reshape(x,(N,1)),np.reshape(y,(N,1)),axis = 1)

	indices = np.arange(0,100,1) # All indices
	validation_index = np.arange(0,100,10) # Indices alloted to validation 
	test_index = np.arange(1,100,5) # Indices alloted to test
	train_index = np.delete(indices,np.append(test_index,validation_index)) # Indices allot to train

	data_train = data[train_index]
	data_validation = data[validation_index]
	data_test = data[test_index]

	m = hyper_parameter_tuning(data_train,data_validation)
	print("Best model has degree: ",m)



if __name__ == "__main__":
	main()