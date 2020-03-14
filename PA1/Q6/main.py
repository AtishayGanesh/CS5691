import numpy as np 
import matplotlib.pyplot as plt

def func(x):
	'''
	Returns the target function applied to the data
	'''
	return np.exp(np.tanh(2*np.pi*x))

def polynomial_regression(data_train,data_test):
	'''
	Performs polynomial regression on the training data and evaluates it on the test data
	'''
	N = 80 # Size of dataset for training
	data_indices = np.arange(0,80,80//N)[:N]
	data = data_train[data_indices] # Part of total training data used for regression

	degrees = [1,3,6,9] # Range of degrees of polynomial to be fit

	# We will perform polynomial regression assuming y = Aw + n. Each column of A will have the observations x raised to the corresponding power
	A_train = np.ones((N,10)) # Matrix for training
	A_test = np.ones((20,10)) # Matrix for testing
	A_plot = np.ones((99,10)) # Matrix for plotting

	y_fit = np.zeros((N,4)) # Will store the output training points
	test_err = np.zeros(4) # Stores the test error for different degrees
	train_err = np.zeros(4) # Stores the train error for different degrees

	# Range of x values to plot the function for
	x_plot = np.linspace(0.001,1,99)
	y_plot = func(x_plot)

	# Generating the A matrix
	for i in range(0,10):
		A_train[:,i] = data[:,0]**i
		A_test[:,i] = data_test[:,0]**i
		A_plot[:,i] = x_plot**i

	# Performing polynomial regression
	for d in range(4):
		A_d = A_train[:,:degrees[d]+1] # Using only the part of A matrix required for the degree under consideration
		w = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_d.T,A_d)),A_d.T),data[:,1]) # Least squares solution
		print("Coefficients for degree {} are {}".format(degrees[d],w))
		y_fit[:,d] = np.matmul(A_d,w) # Points fit

		# Finding errors
		y_test = np.matmul(A_test[:,:degrees[d]+1],w) # Output of model for test data

		# RMS Error
		test_err[d] = np.linalg.norm(data_test[:,1] - y_test)/(20**0.5)
		train_err[d] = np.linalg.norm(data[:,1] - y_fit[:,d])/(N**0.5)

		# Plotting regression output
		plt.title('Plot of target function and the polynomial fit\n Degree = {}'.format(degrees[d]))
		plt.plot(x_plot,func(x_plot),'k',label = "Actual function to be fit")
		plt.plot(x_plot,np.matmul(A_plot[:,:degrees[d]+1],w),label = "Polynomial fit")
		plt.xlabel(r'x $\rightarrow$')
		plt.ylabel(r'y $\rightarrow$')
		plt.ylim([0,4])
		plt.legend()
		plt.grid()
		#plt.savefig('plots/ten_points_degree_{}.jpg'.format(degrees[d]))
		#plt.show()

		# Following code is for the scatter plot of model output vs target output for the best performing model
		
		"""
		if d == 1: # Degree = 3
			plt.clf()
			plt.title('Plot of model output vs target output for training data')
			plt.plot(data[:,1],y_fit[:,d],'ro')
			plt.plot(y_plot,y_plot,'r', label = 'y=x')
			plt.legend()
			plt.xlabel(r'Target Output $\rightarrow$')
			plt.ylabel(r'Model Output $\rightarrow$')
			plt.grid()
			#plt.savefig('plots/scatter_output_training.jpg')
			plt.show()

			plt.clf
			plt.title('Plot of model output vs target output for test data')
			plt.plot(data_test[:,1],y_test,'bo')
			plt.plot(y_plot,y_plot,'r',label = 'y=x')
			plt.legend()
			plt.xlabel(r'Target Output $\rightarrow$')
			plt.ylabel(r'Model Output $\rightarrow$')
			#plt.savefig('plots/scatter_output_test.jpg')
			plt.grid()
			plt.show()
		"""

	# Plotting the data points fit
	plt.title('Plot of polynomials of different degrees fit to data points')
	plt.plot(data[:,0],data[:,1],'bo', label = 'Actual data points with noise')
	plt.plot(data[:,0],y_fit[:,0],'r', label = 'Degree 1')
	plt.plot(data[:,0],y_fit[:,1],'g', label = 'Degree 3')
	plt.plot(data[:,0],y_fit[:,2],'y', label = 'Degree 6')
	plt.plot(data[:,0],y_fit[:,3],'k', label = 'Degree 9')
	plt.ylim([0,4])
	plt.xlabel(r'x $\rightarrow$')
	plt.ylabel(r'y $\rightarrow$')
	plt.legend(loc='lower right')
	plt.grid()
	#plt.savefig('plots/ten_points_all_fit.jpg')
	#plt.show()

	# Plotting the variation of error with degree of polynomial
	plt.cla()
	plt.title('Plot of training and test RMS errors with degree of polynomial fit')
	plt.plot(degrees,test_err,'r', label = 'Test Error')
	plt.plot(degrees,train_err,'b',label = 'Training Error')
	plt.legend()
	plt.grid()
	plt.xlabel(r'Degree of polynomial $\rightarrow$')
	plt.ylabel(r'RMS Error $\rightarrow$')
	#plt.savefig('plots/eighty_points_rms_error.jpg')
	plt.show()

	min_test_err_degrees = degrees[np.argmin(test_err)]

	# Varying the training data size
	plt.clf()
	N_list = [20,40,60,80]
	A_train = np.ones((80,10)) # Matrix for training
	A_test = np.ones((20,10)) # Matrix for testing
	test_err = np.zeros(4)
	train_err = np.zeros(4)

	# Generating the A matrix
	for i in range(0,10):
		A_train[:,i] = data_train[:,0]**i
		A_test[:,i] = data_test[:,0]**i

	# Vary degree of polynomial from 1 to 9
	for d in range(4):
		plt.title("Polynomials of degree {} fit for varying training data sizes".format(degrees[d]))
		plt.plot(data[:,0],y_fit[:,d],label = "Data samples = 10")
		# Change the data set size
		for n in range(4):
			data_indices = np.arange(0,80,80//N_list[n])[:N_list[n]]
			data_n = data_train[data_indices] # Part of total training data used for regression
			A_train_n = A_train[data_indices,:degrees[d]+1] # Part of A matrix required
			w = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_train_n.T,A_train_n)),A_train_n.T),data_n[:,1]) # Least Squares Solution
			y_fit_n = np.matmul(A_train_n,w)

			# Finding errors
			test_err[n] = np.linalg.norm(data_test[:,1] - np.matmul(A_test[:,:degrees[d]+1],w))/(20**0.5)
			train_err[n] = np.linalg.norm(data_n[:,1] - y_fit_n)/(N_list[n]**0.5)

			plt.plot(data_n[:,0],y_fit_n, label = "Data samples = {}".format(N_list[n]))
		plt.plot(x_plot,func(x_plot),'k',label = "Actual function to be fit")
		plt.legend(loc='lower right')
		plt.xlabel(r'x $\rightarrow$')
		plt.ylabel(r'y $\rightarrow$')
		plt.grid()
		#plt.savefig('plots/fit_varying_data_size_deg_{}'.format(degrees[d]))
		#plt.show()

		"""
		plt.title("Variation of RMS Error with different training data sizes for polynomial of degree {}".format(N_list[n]))
		plt.plot(N_list,test_err, label = "Test Error")
		plt.plot(N_list,train_err, label = "Training Error")
		plt.legend()
		plt.xlabel(r'Number of samples $\rightarrow$')
		plt.ylabel(r'RMS Error $\rightarrow$')
		plt.show()
		"""

	return min_test_err_degrees

def main():
	# Generating the data
	N = 100
	x = np.arange(101)[1:]/101 # Generating 100 samples between 0 and 1
	y = func(x) + np.random.normal(0,0.2**0.5,N) # Adding Gaussian noise to the data 
	data = np.append(np.reshape(x,(N,1)),np.reshape(y,(N,1)),axis = 1)

	# Dividing into training and test data
	indices = np.arange(0,100,1) # All indices
	test_index = np.arange(1,100,5) # Indices alloted to test
	train_index = np.delete(indices,test_index) # Indices alloted to train
	data_train = data[train_index]
	data_test = data[test_index]

	m = polynomial_regression(data_train,data_test)
	print("Best model (least test error) has degree: ",m)

if __name__ == "__main__":
	main()