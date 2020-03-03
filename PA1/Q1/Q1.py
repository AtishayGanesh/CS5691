import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Bayes_Classifier:
    def __init__(self, dataset_number=1, model_number=1):
        self.dataset, self.Class_Points = self.data_preprocessing(dataset_number)
        self.prior = self.calculate_prior()
        self.test_split = 0.2
        self.split_data()
        self.estimate_CCD(model_number)

    def loadData(self, dataset_number):
        ''' Reads the csv file'''
        df = pd.read_csv('Dataset_{}_Team_6.csv'.format(dataset_number))
        val = df.values
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
        return prior

    def estimate_CCD(self, model_number):
        '''
        Estimates the class conditional density, given the model
        '''
        self.means = self.estimate_means()
        if model_number == 1:
            self.covariances = [np.eye(2) for i in range(3)]
            
        if model_number == 2:
            sum_sq = np.zeros((1,2))
            l = 0 
            for i in range(3):
                l += len(self.train_data[i])
                sum_sq += np.sum(np.square(self.train_data[i]-self.means[i]),axis=0)
            self.covariances = [np.diagflat(sum_sq/l)]*3
        if model_number == 3:
            self.covariances = []
            for i in range(3):
                sum_sq = np.zeros((1,2))
                l = len(self.train_data[i])
                sum_sq = np.sum(np.square(self.train_data[i]-self.means[i]),axis=0)
                (sum_sq/l)
                self.covariances.append(np.diagflat(sum_sq/l))
        if model_number == 4:
            sum_sq = np.zeros((2,2))
            l = 0 
            for i in range(3):
                l += len(self.train_data[i])
                normed = (self.train_data[i]-self.means[i])
                sum_sq += (normed.T@normed)
            self.covariances = [(sum_sq/l)]*3

        if model_number == 5:
            self.covariances = []
            for i in range(3):
                l = len(self.train_data[i])
                normed = (self.train_data[i]-self.means[i])

                sum_sq = normed.T@normed
                self.covariances.append((sum_sq/l))


        print(self.covariances,'\n')


    def estimate_means(self):
        '''
        Estimates the mean of the datasets
        '''
        mean = ([np.mean(self.train_data[i], axis=0) for i in range(3)])
        return mean




if __name__ == '__main__':
    for i in range(1,6):
        b = Bayes_Classifier(1,i)
