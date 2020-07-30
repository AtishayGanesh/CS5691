# CS5691: PRML Programming Assignment 2 - Part 1.1: Kernel Ridge Regression
# Roll Numbers: EE17B102 & EE17B155

import numpy as np
import pandas as pd
from sklearn import kernel_ridge

def poly_kernel(A,B,c=0):
	fact = np.matmul(A.T,B) + c
	return fact**2

def linear_kernel(A,B,c=0):
	return np.matmul(A.T,B) + c

def kernel_regression(train_data,kernel_index,test_data,lambd = 1,c = 0):
	"""
	train_data : training data
	kernel_index : 0 (linear) or 1 (polynomial)
	test_data: test data
	lambd : regularization parameter
	c : offset parameter in the kernel
	"""
	n_dim = 13 # Number of given features
	X = train_data.T[:n_dim,:]
	X = np.append(X,np.ones((1,X.shape[1])),0) # Adding bias
	Y = train_data.T[n_dim,:]

	X_test = test_data.T[:n_dim,:]
	X_test =  np.append(X_test,np.ones((1,X_test.shape[1])),0) # Adding bias
	Y_test = test_data.T[n_dim,:]

	kernel = poly_kernel if kernel_index else linear_kernel
	kernel_name = "poly" if kernel_index else "linear"

	# Kernel Ridge Regression
	XXT = kernel(X,X,c)
	pseudo_inv = np.linalg.inv(XXT + lambd*np.eye(XXT.shape[0]))	
	Y_pred_train = np.matmul(kernel(X,X,c),np.matmul(pseudo_inv,Y))
	Y_pred_test = np.matmul(kernel(X_test,X,c),np.matmul(pseudo_inv,Y))

	# Calculating Mean Square Error
	train_loss = np.sum((Y_pred_train - Y)**2)/np.shape(Y)[0]
	test_loss = np.sum((Y_pred_test - Y_test)**2)/np.shape(Y_test)[0]
	print("{} kernel: train_error:{}  test_error:{}".format(kernel_name,train_loss,test_loss))
	
def main():
	data = np.array(pd.read_csv('PA2_Datasets/Regression_dataset.csv'))
	
	# Splitting data in ratio 4:1 for train:test
	n_train = int(0.8*data.shape[0])
	train_data = data[:n_train,:]
	test_data = data[n_train:,:]

	kernel_regression(train_data,0,test_data,20,1) # Linear Kernel
	kernel_regression(train_data,1,test_data,0.1,1) # Polynomial Kernel

if __name__ == "__main__":
	main()