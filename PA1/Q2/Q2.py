import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Bayesian_Estimation():
    def __init__(self,sample_size,rates,plot=False):
        self.data_preprocessing(plot)
        for r in rates:
            vals = []
            for s in sample_size:
                np.random.shuffle(self.dataset)
                vals.append(self.classify(s,r))
                print()

            [plt.plot(val[0],val[1]) for val in vals]
            plt.title(
                "Distribution with various sample size and dogmatism = {}".format(r))
            plt.xlabel("x value")
            plt.ylabel("Probability")
            plt.legend(sample_size)
            
            plt.show()



    def loadData(self):
        ''' Reads the csv file'''
        df = pd.read_csv('Dataset_3_Team_6.csv')
        val = df.values
        return val[:,0]

    def data_preprocessing(self,plot=False):
        '''
        Calls the function that loads the dataset, and splits data by class
        '''
        self.dataset = self.loadData()
        if plot:
            self.plot()

    def plot(self):
        print(np.mean(self.dataset))
        plt.hist(self.dataset,bins=100,density=True)
        plt.title("Distribution of dataset 3")
        plt.ylabel("Probability of occurences")
        plt.xlabel("Value")
        plt.plot()
        plt.show()

    def classify(self,sample_size,rate):
        data =  self.dataset[0:sample_size]
        xn = np.mean(data)
        u0 = -1
        un = (u0*rate+ xn*sample_size)/(sample_size+rate)
        sigma = np.sum(np.square(data-un))/sample_size
        sigma_n = sigma/(sample_size+rate)
        print("With sample size = {}, and Dogmatism = {},{} = {}".format(
            sample_size,rate,chr(963),np.sqrt(sigma)))
        sigma_posterior = sigma+ sigma_n
        x = np.linspace(
            un - 3*np.sqrt(sigma_posterior), un + 3*np.sqrt(sigma_posterior), 150)
        y = (1/np.sqrt(2*np.pi*sigma_posterior))*(
            np.exp(-np.square(x-un)/(2*sigma_posterior)))
        return(x,y)





if __name__ == '__main__':
    sample_size =[10,100,1000]
    rates = [0.1,1,10,100]
    B = Bayesian_Estimation(sample_size,rates)



