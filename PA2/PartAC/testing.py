import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

def poly_kernel(A,B,c=0):
	fact = np.matmul(A.T,B) + c
	return fact**2

def linear_kernel(A,B,c=0):
	return np.matmul(A.T,B) + c

def kernel_regression(train_data,kernel_index,test_data,lambd = 1,c = 0):

	kernel = poly_kernel if kernel_index else linear_kernel
	kernel_name = "poly" if kernel_index else "linear"

	n_dim = 13 # Number of given features
	X = train_data.T[:n_dim,:]
	X = np.append(X,np.ones((1,X.shape[1])),0) # Adding bias
	Y = train_data.T[n_dim,:]

	X_test = test_data.T[:n_dim,:]
	X_test =  np.append(X_test,np.ones((1,X_test.shape[1])),0) # Adding bias
	Y_test = test_data.T[n_dim,:]

	clf = KernelRidge(alpha = 0.5, kernel = "poly")
	clf.fit(X.T,Y)

	Y_pred_train = clf.predict(X.T)
	Y_pred_test = clf.predict(X_test.T)

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
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def fit_data_svm(train_data,test_data,kernel_index,C = 1,coef0 = 0):
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

	clf = LogisticRegression(random_state = 0).fit(X,Y)

	Y_pred_train = clf.predict(X)
	Y_pred_test = clf.predict(X_test)

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
"""
"""
from sklearn.linear_model import Perceptron
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
	kernel = poly_kernel if kernel_index else linear_kernel
	kernel_name = "polynomial" if kernel_index else "linear"
	X = train_data[:,:2]
	X = np.append(X,np.ones((X.shape[0],1)),1) # Adding bias
	Y = 2*(train_data.T[2,:]-0.5) # Mapping classes 0 and 1 to -1 and 1 respectively
	X_test = test_data[:,:2]
	X_test = np.append(X_test,np.ones((X_test.shape[0],1)),1) # Adding bias

	clf = Perceptron(tol = 1e-3,random_state = 0)

	clf.fit(X,Y)
	# Final predictions mapped to 0 and 1
	Y_pred_train = clf.predict(X)/2 + 0.5
	Y_pred_test = clf.predict(X_test)/2 + 0.5

	# Calculate accuracy
	train_loss = np.sum(np.abs(Y_pred_train - train_data.T[2,:]))/np.shape(train_data.T[2,:])
	test_loss = np.sum(np.abs(Y_pred_test - test_data.T[2,:]))/np.shape(test_data.T[2,:])
	print("Perceptron {} kernel: training accuracy:{} test accuracy:{} epochs:{}".format(kernel_name,1-train_loss[0],1-test_loss[0],1))
	print("Total number of updates: {} Theoretical Bound: {}\n".format(1,1))


def main():
	dataset_num = 1 # Use either dataset 1 or dataset 3
	data = np.array(pd.read_csv('PA2_Datasets/Dataset_{}/Dataset_{}_Team_8.csv'.format(dataset_num,dataset_num)))
	n_train = int(0.8*data.shape[0]) # Split data in 4:1 for train:test
	train_data = data[:n_train,:]
	test_data = data[n_train:,:]

	perceptron(train_data,test_data,1) # Polynomial Kernel
	perceptron(train_data,test_data,0) # Linear Kernel

if __name__ == "__main__":
	main()
"""