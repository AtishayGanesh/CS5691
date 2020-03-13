import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as cnf

class Bayes_Classifier:
    def __init__(self, dataset_number=1, model_number=1):
        self.dataset_number = dataset_number
        self.model_number= model_number
        self.dataset, self.Class_Points = self.data_preprocessing(dataset_number)
        self.prior = self.calculate_prior()
        self.test_split = 0.2
        self.split_data()
        self.means, self.covariances = self.estimate_CCD(model_number)
        self.loss_matrix = np.array([[0,2,1],[2,0,3],[1,3,0]])
        self.train_accr,self.train_pred = self.estimate_posteriors()
        self.test_accr,self.test_pred = self.estimate_posteriors(test=True)
        #if (dataset_number ==2 and model_number==5) or (dataset_number==1 and model_number==3):
        

    def loadData(self, dataset_number):
        ''' Reads the csv file'''
        df = pd.read_csv('Dataset_{}_Team_6.csv'.format(dataset_number))
        val = df.values
        self.min = np.amin(val,axis=0)[0:2]
        self.max = np.amax(val,axis=0)[0:2]
        return val

    def data_preprocessing(self, dataset_number):
        '''
        Calls the function that loads the dataset, and splits data by class
        '''
        dataset = self.loadData(dataset_number)
        Class_Points = [dataset[:, 0:2][np.where(dataset[:, 2] == i)] for i in range(3)]
        return dataset, Class_Points

    def split_data(self):
        '''
        Split train and test data
        '''
        self.train_data = []
        self.test_data = []
        for i in range(3):
            l = len(self.Class_Points[i])
            np.random.shuffle(self.Class_Points[i])
            self.train_data.append(self.Class_Points[i][:int((1-self.test_split)*l)])
            self.test_data.append(self.Class_Points[i][int((1-self.test_split)*l):])


    def calculate_prior(self):
        '''
        Calculates Prior given the dataset
        '''
        prior = np.array([len(self.Class_Points[i])/len(self.dataset) for i in range(3)])
        #print(prior)
        return prior

    def estimate_CCD(self, model_number):
        '''
        Estimates the class conditional density, given the model
        '''
        means = self.estimate_means()
        if model_number == 1:
            covariances = [np.eye(2) for i in range(3)]
            
        if model_number == 2:
            sum_sq = np.zeros((1,2))
            l = 0 
            for i in range(3):
                l += len(self.train_data[i])

                sum_sq += np.sum(np.square(self.train_data[i]-means[i]),axis=0)
            covariances = [np.diagflat(sum_sq/l)]*3
        if model_number == 3:
            covariances = []
            for i in range(3):
                sum_sq = np.zeros((1,2))
                l = len(self.train_data[i])
                sum_sq = np.sum(np.square(self.train_data[i]-means[i]),axis=0)
                (sum_sq/l)
                covariances.append(np.diagflat(sum_sq/l))
        if model_number == 4:
            sum_sq = np.zeros((2,2))
            l = 0 
            for i in range(3):
                l += len(self.train_data[i])
                normed = (self.train_data[i]-means[i])
                sum_sq += (normed.T@normed)
            covariances = [(sum_sq/l)]*3

        if model_number == 5:
            covariances = []
            for i in range(3):
                l = len(self.train_data[i])
                normed = (self.train_data[i]-means[i])
                sum_sq = normed.T@normed
                covariances.append((sum_sq/l))

        return (means,covariances)

    def estimate_posteriors(self,test=False,new_data =None):
        '''Estimating the posterior'''
        self.posteriors = []
        self.likelihoods = []
        class_num = 3 if new_data is None else 1
        for i in range(class_num):
            if new_data is not None:
                val = new_data
            elif test:
                val = self.test_data[i]
            else:
                val = self.train_data[i]
            log_likelihood =  [np.array([-0.5*(
                val[k]- self.means[j])@np.linalg.inv(
                self.covariances[j])@((
                    val[k]- self.means[j]).T)-0.5*np.log(np.linalg.det(
            self.covariances[j])) for k in range(len(val))]) for j in range(3)]

            posterior = np.array([ log_likelihood[j]+np.log(self.prior[j]) for j in range(3)])
            if self.model_number==1:
                self.posteriors.append((posterior.T))    
            else:
                self.posteriors.append(np.exp(posterior.T))
        
        if self.model_number==1:
            self.bayes = [-self.posteriors[j] for j in range(class_num)]
        else:
            self.bayes = [self.posteriors[j]@self.loss_matrix for j in range(class_num)]

        if new_data is not None:
            return np.argmin(self.bayes[0],axis=1),(
                np.exp(np.array([log_likelihood[j] for j in range(3)]   ))/(2*np.pi))
        else:
            accr = 1-(np.count_nonzero(
                np.argmin(self.bayes[0],axis=1))+np.count_nonzero(
                np.argmin(self.bayes[1],axis=1)-1)+np.count_nonzero(
                np.argmin(self.bayes[2],axis=1)-2))/(len(
                self.bayes[0]) +len(self.bayes[1])+len(self.bayes[2]))
        txt = 'Testing' if test else "Training" 

        print("Dataset {} , Model {}, {} Accuracy {}".format(
           self.dataset_number,self.model_number,txt,accr))

        return accr,np.concatenate((np.argmin(
            self.bayes[0],axis=1),np.argmin(
            self.bayes[1],axis=1),np.argmin(self.bayes[2],axis=1)))
        
    def estimate_means(self):
        '''
        Estimates the mean of the datasets
        '''
        mean = ([np.mean(self.train_data[i], axis=0) for i in range(3)])
        return mean

    def display_decision_surface(self,n_pts):
        mi = min(self.min[0],self.min[1])
        ma = max(self.max[0],self.max[1])
        x1 = np.linspace(mi, ma,n_pts)
        x2 = np.linspace(mi, ma,n_pts)
        x1v,x2v = np.meshgrid(x1,x2)
        op_data,op_levels = self.estimate_posteriors(
            new_data = np.stack((x1v.flatten(),x2v.flatten()),axis=-1 ))
        cs = plt.contourf(x1,x2,op_data.reshape(n_pts,n_pts),2)
        X1 = np.split(self.train_data[0],2,axis=1)
        X2 = np.split(self.train_data[1],2,axis=1)
        X3 = np.split(self.train_data[2],2,axis=1)
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
            for pc in cs.collections]
        c1, = plt.plot(X1[0],X1[1],'ro')
        c2, = plt.plot(X2[0],X2[1],'bx')
        c3, = plt.plot(X3[0],X3[1],'gd')
        proxy += [c1,c2,c3]
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Decision Surface with training data overlayed")
        plt.legend(proxy,["Class 0","Class 1","Class 2","Train 0","Train 1","Train 2"])      
        plt.xlim((self.min[0],self.max[0]))
        plt.ylim((self.min[1],self.max[1]))
        plt.show()
        return op_levels,x1,x2

    def display_contour_curves(self,op_levels,n_pts,x1,x2):

        plt.contour(x1,x2,op_levels[0].reshape(n_pts,n_pts),4)
        plt.contour(x1,x2,op_levels[1].reshape(n_pts,n_pts),4)
        plt.contour(x1,x2,op_levels[2].reshape(n_pts,n_pts),4)
        plt.title("Constant Density Curves and Eigenvectors")

        w1,v1 = np.linalg.eig((self.covariances[0]))
        w2,v2 = np.linalg.eig((self.covariances[1]))
        w3,v3 = np.linalg.eig((self.covariances[2]))

        means_x = (self.means[0][0],self.means[0][0],
            self.means[1][0],self.means[1][0],self.means[2][0],self.means[2][0])
        means_y = (self.means[0][1],self.means[0][1],
            self.means[1][1],self.means[1][1],self.means[2][1],self.means[2][1])

        u1 = np.asarray([v1[:,0],v1[:,1]]+[v2[:,0],v2[:,1]]+[v3[:,0],v3[:,1]])
        plt.quiver(means_x,means_y,u1[:,0],u1[:,1],scale=20,headwidth=2,headlength=4)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

    def display_results(self):
        train_pred = self.train_pred
        test_pred = self.test_pred
        train_act = np.concatenate((np.zeros(len(
            self.train_data[0])),np.ones(len(
            self.train_data[1])),2*np.ones(len(self.train_data[2]))))
        test_act = np.concatenate((np.zeros(len(
            self.test_data[0])),np.ones(len(
            self.test_data[1])),2*np.ones(len(self.test_data[2]))))
        print(cnf(train_act,train_pred))
        print(cnf(test_act,test_pred))

        n_pts=500
        op_levels,x1,x2 = self.display_decision_surface(n_pts)
        self.display_contour_curves(op_levels,n_pts,x1,x2)

        







if __name__ == '__main__':
    d1 = []
    d2 = []
    for i in range(1,6):
        d1.append(Bayes_Classifier(1,i))
    for i in range(1,6):
        d2.append(Bayes_Classifier(2,i))
    r1 = np.asarray([s.test_accr for s in d1])
    r2 = np.asarray([s.test_accr for s in d2])
    d1[b1].display_results()
    d2[4].display_results()


