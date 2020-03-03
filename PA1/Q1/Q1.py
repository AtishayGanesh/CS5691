import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Bayes_Classifier:
    def __init__(self,dataset_number=1):
        self.dataset, self.Class_Points = self.data_preprocessing(dataset_number)
        self.prior = self.calculate_prior()
        self.esti

    def loadData(self,dataset_number):
        df = pd.read_csv('Dataset_{}_Team_6.csv'.format(dataset_number))
        val = df.values
        return val

    def data_preprocessing(self,dataset_number):
        dataset = self.loadData(dataset_number)
        Class_Points = [dataset[:,0:2][np.where(dataset[:,2]==i)] for i in range(3)]
        return dataset,Class_Points

    def calculate_prior(self):
        prior = np.array([len(self.Class_Points[i])/len(self.dataset) for i in range(3)])
        print(prior)
        return prior
        



if __name__ == '__main__':
    b = Bayes_Classifier(1)
