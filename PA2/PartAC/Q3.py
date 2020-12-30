# CS5691: PRML Programming Assignment 2 - Part C: Perceptron Based Classifier
# Roll Numbers: EE17B102 & EE17B155

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

thresh = 1e-10
max_epoch = 1000
actual_alphas = 0

def poly_kernel(A,B,c=0):
	fact = np.matmul(A,B.T) + c
	return fact**2

def linear_kernel(A,B,c=0):
	return np.matmul(A,B.T) + c

def perceptron(train_data,test_data,kernel_index,C = 0):
	"""
	train_data : training data
	test_data : test data
	kernel_index : 0(linear kernel) or 1(polynomial kernel)
	C : Offset in kernel
	""" 
	kernel = poly_kernel if kernel_index else linear_kernel
	kernel_name = "polynomial" if kernel_index else "linear"
	X = train_data[:,:2]
	X = np.append(X,np.ones((X.shape[0],1)),1) # Adding bias
	Y = 2*(train_data.T[2,:]-0.5) # Mapping classes 0 and 1 to -1 and 1 respectively
	X_test = test_data[:,:2]
	X_test = np.append(X_test,np.ones((X_test.shape[0],1)),1) # Adding bias

	# Perceptron Algorithm
	epoch = 0 # Keeps track of number of epochs
	rat = 100 # Keerps track of change in error
	err = 100
	T = X.shape[0]
	alphas = np.zeros(T)
	num_updates = 0
	while(rat>thresh and epoch<max_epoch):
		err_old = err
		loss = 0
		flag = 0
		for i in range(T):
			if np.sign(np.matmul(Y*alphas,kernel(X,X[i,:],C)))!= Y[i]: # If prediction is wrong
				alphas[i] += 1 # Update weights
				num_updates += 1 
				flag = 1
				loss+=1
		err = loss/T # Average loss
		rat = abs(err-err_old)
		epoch += 1
		if flag == 0: # No updates were made, epoch isn't counted
			epoch -= 1

	# Final predictions mapped to 0 and 1
	Y_pred_train = np.sign(np.matmul(kernel(X,X,C),Y*alphas))/2 + 0.5
	Y_pred_test = np.sign(np.matmul(kernel(X_test,X,C),Y*alphas))/2 + 0.5

	# For theoretical bound
	x_norm = []
	m = []
	for i in range(T):
		x_norm.append(np.sqrt(kernel(X[i,:],X[i,:])))#x_norm.append(np.linalg.norm(X[i,:]))
		m.append(Y[i]*(np.matmul(Y*alphas,kernel(X,X[i,:],C)))/np.matmul(np.matmul(Y*alphas,kernel(X,X)),Y*alphas)**0.5)
	r = np.array(x_norm).max()
	rho = np.array(m).min()
	bound = (r/rho)**2


	# Calculate accuracy
	train_loss = np.sum(np.abs(Y_pred_train - train_data.T[2,:]))/np.shape(train_data.T[2,:])
	test_loss = np.sum(np.abs(Y_pred_test - test_data.T[2,:]))/np.shape(test_data.T[2,:])
	print("Perceptron {} kernel: training accuracy:{} test accuracy:{} epochs:{}".format(kernel_name,1-train_loss[0],1-test_loss[0],epoch))
	print("Total number of updates: {} Theoretical Bound: {}\n".format(num_updates,bound))

	# Plot decision boundary
	ax = plt.gca()
	x1_index = np.where(Y == 1)
	x0_index = np.where(Y == -1)
	plt.scatter(X[x1_index,0],X[x1_index,1],color = 'r',label = 'Class 1')
	plt.scatter(X[x0_index,0],X[x0_index,1],color = 'b',label = 'Class 0')
	xx = np.linspace(X[:,0].min(),X[:,0].max(),200)
	yy = np.linspace(X[:,1].min(),X[:,1].max(),200)
	YY,XX = np.meshgrid(yy,xx)
	xy = np.vstack([XX.ravel(),YY.ravel()]).T
	xy = np.append(xy,np.ones((xy.shape[0],1)),1)
	Z = np.sign(np.matmul(kernel(xy,X,C),Y*alphas)).reshape(XX.shape)
	ax.contour(XX,YY,Z,colors = 'k', levels = [0],linestyles = ['-'])
	plt.xlabel('X_1')
	plt.ylabel('X_2')
	plt.legend()
	plt.show()


def main():
	dataset_num = 3 # Use either dataset 1 or dataset 3
	data = np.array(pd.read_csv('PA2_Datasets/Dataset_{}/Dataset_{}_Team_8.csv'.format(dataset_num,dataset_num)))
	n_train = int(0.8*data.shape[0]) # Split data in 4:1 for train:test
	train_data = data[:n_train,:]
	test_data = data[n_train:,:]

	perceptron(train_data,test_data,1) # Polynomial Kernel
	perceptron(train_data,test_data,0) # Linear Kernel

if __name__ == "__main__":
	main()