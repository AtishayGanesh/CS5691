# CS5691: PRML Programming Assignment 2 - Part 1.2: Kernel Logistic Regression and Kernel SVM
# Roll Numbers: EE17B102 & EE17B155

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn import preprocessing

def fit_data_svm(train_data,test_data,kernel_index,C = 1,coef0 = 0):
	"""
	train_data : training data
	test_data : test data
	kernel_index : 0 (linear kernel) or 1 (polynomial kernel)
	C : Regularization parameter
	coef0 : Offset in kernel (only for poly)
	"""
	kernel = "poly" if kernel_index else "linear"
	clf = svm.SVC(kernel = kernel,degree = 2, C = C,coef0 = coef0,break_ties = True)
	X = train_data[:,:2]
	X = np.append(X,np.ones((X.shape[0],1)),1) # Adding bias
	Y = train_data.T[2,:]
	X_test = test_data[:,:2] 
	X_test = np.append(X_test,np.ones((X_test.shape[0],1)),1) # Adding bias

	# Encoding Y for fit()
	lab_enc = preprocessing.LabelEncoder()
	encoded_Y = lab_enc.fit_transform(Y)

	# Fit SVM
	clf.fit(X,encoded_Y)
	Y_pred_train = clf.predict(X)
	Y_pred_test = clf.predict(X_test)

	# Calculate and print accuracy
	train_loss = np.sum(np.abs(Y_pred_train - train_data.T[2,:]))/np.shape(train_data.T[2,:])
	test_loss = np.sum(np.abs(Y_pred_test - test_data.T[2,:]))/np.shape(test_data.T[2,:])
	print("SVM {}: training accuracy:{} test accuracy:{}".format(kernel,1-train_loss[0],1-test_loss[0]))

	# Plotting Decision Boundary
	ax = plt.gca()
	x1_index = np.where(Y == 1)
	x0_index = np.where(Y == 0)
	plt.scatter(X[x1_index,0],X[x1_index,1],color = 'r',label = 'Class 1 Non-Support Vectors')
	plt.scatter(X[x0_index,0],X[x0_index,1],color = 'b',label = 'Class 0 Non-Support Vectors')
	xx = np.linspace(X[:,0].min(),X[:,0].max(),30)
	yy = np.linspace(X[:,1].min(),X[:,1].max(),30)
	YY,XX = np.meshgrid(yy,xx)
	xy = np.vstack([XX.ravel(),YY.ravel()]).T
	xy = np.append(xy,np.ones((xy.shape[0],1)),1)
	Z = clf.decision_function(xy).reshape(XX.shape)
	support_indices = clf.support_
	ax.contour(XX,YY,Z,colors = 'k', levels = [-1,0,1],linestyles = ['--','-','--'])
	x1_index = np.where(Y[support_indices]==1)
	x0_index = np.where(Y[support_indices]==0)
	ax.scatter(clf.support_vectors_[x1_index,0],clf.support_vectors_[x1_index,1],color = 'purple',edgecolors = 'k',label = 'Class 0 Support Vectors')
	ax.scatter(clf.support_vectors_[x0_index,0],clf.support_vectors_[x0_index,1],color = 'g',edgecolors = 'k',label = 'Class 1 Support Vectors')
	plt.xlabel('X_1')
	plt.ylabel('X_2')
	plt.legend()
	plt.show()

def poly_kernel(A,B,c=0):
	fact = np.matmul(A,B.T) + c
	return fact**2

def linear_kernel(A,B,c=0):
	return np.matmul(A,B.T) + c

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def fit_data_lg(train_data,test_data,kernel_index,C = 0):
	"""
	train_data : training data
	test_data : test data
	kernel_index : 0 (linear kernel) or 1 (polynomial kernel)
	C : Offset in kernel
	"""
	kernel = poly_kernel if kernel_index else linear_kernel
	kernel_name = "poly" if kernel_index else "linear"
	X = train_data[:,:2]
	X = np.append(X,np.ones((X.shape[0],1)),1) # Adding bias
	Y = train_data[:,2]
	X_test = test_data[:,:2]
	X_test = np.append(X_test,np.ones((X_test.shape[0],1)),1) # Adding bias

	thresh = 1e-3 # Iterate till change in weight vector is less than thresh
	change = 1e-3
	alpha = np.random.rand(X.shape[0]) # Initial random initialisation 
	K = kernel(X,X,C) 

	# Iterate until convergence
	while (change>=thresh): 
		AW = np.matmul(K,alpha)
		Z = sigmoid(AW)
		R = np.diag(Z*(1-Z))
		B = AW - np.matmul(np.linalg.inv(R+1e-6*np.eye(X.shape[0])),Z-Y) # Need to add a minimum value to R incase it is not invertible
		alpha_updated = np.matmul(np.matmul(R,np.linalg.inv(np.matmul(K,R)+1e-6*np.eye(X.shape[0]))),B) # Need to add a minimum value to R in case it is not invertible
		change = np.linalg.norm(alpha_updated - alpha)/alpha.shape[0]
		alpha = alpha_updated*1

	# Make final predictions
	Y_pred_train = sigmoid(np.matmul(K,alpha))
	Y_pred_test = sigmoid(np.matmul(kernel(X_test,X,C),alpha))

	# If the output of the model is >0.5 we will predict class1, else predict class0
	threshold = 0.5
	Y_pred_test[np.where(Y_pred_test>=threshold)] = 1
	Y_pred_test[np.where(Y_pred_test<threshold)] = 0
	Y_pred_train[np.where(Y_pred_train>=threshold)] = 1
	Y_pred_train[np.where(Y_pred_train<threshold)] = 0

	# Calculate and print accuracy
	train_loss = np.sum(np.abs(Y_pred_train - train_data.T[2,:]))/np.shape(train_data.T[2,:])
	test_loss = np.sum(np.abs(Y_pred_test - test_data.T[2,:]))/np.shape(test_data.T[2,:])
	print("Logistic Regression {}: training accuracy:{} test accuracy:{}".format(kernel_name,1-train_loss[0],1-test_loss[0]))

	# Plot Decision Boundary
	ax = plt.gca()
	x1_index = np.where(Y == 1)
	x0_index = np.where(Y == 0)
	plt.scatter(X[x1_index,0],X[x1_index,1],color = 'r',label = 'Class 1')
	plt.scatter(X[x0_index,0],X[x0_index,1],color = 'b',label = 'Class 0')
	xx = np.linspace(X[:,0].min(),X[:,0].max(),100)
	yy = np.linspace(X[:,1].min(),X[:,1].max(),100)
	YY,XX = np.meshgrid(yy,xx)
	xy = np.vstack([XX.ravel(),YY.ravel()]).T
	xy = np.append(xy,np.ones((xy.shape[0],1)),1)
	Z = sigmoid(np.matmul(kernel(xy,X,C),alpha)).reshape(XX.shape)
	ax.contour(XX,YY,Z,colors = 'k', levels = [0.5],linestyles = ['-'])
	plt.xlabel('X_1')
	plt.ylabel('X_2')
	plt.legend()
	plt.show()

def main():
	data = np.array(pd.read_csv('PA2_Datasets/Dataset_3/Dataset_3_Team_8.csv'))
	n_train = int(0.8*data.shape[0]) # Split data in ratio 4:1 for train:test
	train_data = data[:n_train,:]
	test_data = data[n_train:,:] 
	
	fit_data_svm(train_data,test_data,0) # SVM with linear kernel
	fit_data_svm(train_data,test_data,1) # SVM with polynomial kernel
	fit_data_lg(train_data,test_data,0,0) # Logistic Regression with linear kernel
	fit_data_lg(train_data,test_data,1,10) # Logistic Regression with polynomial kernel

if __name__ == "__main__":
	main()