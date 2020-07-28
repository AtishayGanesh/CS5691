# Use dataset with weighted ratings and remarks

import pickle
import numpy as np
import statistics
import sklearn
import sklearn.svm
import csv
import sklearn.ensemble
import sklearn.naive_bayes
from datetime import datetime
import argparse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def return_data(train_data,test=False):
    epoch = datetime.utcfromtimestamp(0)
    keys = train_data.keys()
    ratings = []
    ratings_weighted = []
    ratings2 = []
    avg_remarks =[]
    avg_remarks_accept = []
    remarks_accept_ratio = []
    std_remarks =[]
    std_ratings =[]
    left = []
    comp = []
    date = []
    ct0= 0

    for k in keys:
        l = train_data[k]['rating']

        ratings.append(np.mean(l))
        #ratings_weighted.append(np.mean(train_data[k]['rating_weighted']))
        ratings2.append(np.average(np.log(l)))

        std_ratings.append(np.mean((l - np.mean(l))**2)**0.5)

        company_name = np.zeros(len(compnames))
        company_name[compnames.index(train_data[k]['comp'])] = 1.0
        comp.append(compnames.index(train_data[k]['comp']))
        date.append((train_data[k]['lastdate'] - epoch).days)

        if len(train_data[k]['remarks']) ==0:
            #average length is 89.14, so for those who dont have any comments, just putting average length
            avg_remarks.append(89.14)
            avg_remarks_accept.append(2)
            std_remarks.append(1)
            remarks_accept_ratio.append(2)
            ct0 +=1
        else:
            #print(np.mean([train_data[k]['remarks'][j][-1] for j in train_data[k]['remarks'].keys()]))
            if not np.mean([train_data[k]['remarks'][j][-1] for j in train_data[k]['remarks'].keys()]):
                avg_remarks_accept.append(2)
            else:
                #print("x")
                avg_remarks_accept.append(statistics.mean([train_data[k]['remarks'][j][-2] for j in train_data[k]['remarks'].keys()]))
            std_remarks.append((statistics.mean([np.abs(np.log(1+train_data[k]['remarks'][j][-1])) for j in train_data[k]['remarks'].keys()])))
            avg_remarks.append(statistics.mean([train_data[k]['remarks'][j][0] for j in train_data[k]['remarks'].keys()]))
            remarks_accept_ratio.append(statistics.mean([train_data[k]['remarks'][j][-1] for j in train_data[k]['remarks'].keys()]))

        if test==False:
            left.append(train_data[k]['left'])

    if test==False:
        left = np.array(left)
    ratings = (np.array(ratings)) 
    ratings = (ratings-np.average(ratings))/np.std(ratings)
    ratings2 = (np.array(ratings2)) 
    ratings2 = (ratings2-np.average(ratings2))/np.std(ratings2)
    remarks_accept_ratio = np.array(remarks_accept_ratio)
    std_ratings = np.array(std_ratings)
    std_ratings = (std_ratings-np.average(std_ratings))/np.std(std_ratings)
    avg_remarks = np.array(avg_remarks)
    avg_remarks = (avg_remarks-np.average(avg_remarks))/np.std(avg_remarks)
    avg_remarks_accept = np.array(avg_remarks_accept)
    avg_remarks_accept = (avg_remarks_accept-np.average(avg_remarks_accept))/np.std(avg_remarks_accept)
    comp =np.array(comp)
    comp = (comp-np.average(comp))/np.std(comp)
    std_remarks =np.array(std_remarks)
    std_remarks = (std_remarks-np.average(std_remarks))/np.std(std_remarks)
    date = np.array(date)
    date = (date-np.average(date))/np.std(date)
    
    #x = (np.stack([date,comp],-1))

    x = (np.stack([avg_remarks_accept,ratings,date,comp,remarks_accept_ratio],-1))
    #x = (np.stack([ratings,ratings2,std_ratings,avg_remarks,avg_remarks_accept,std_remarks,date,comp],-1))

    if test== False:
        return x,left
    else:
        return x,keys

def main(xt,yt,x_test,y_test,emp,compnames,test=False):
    GBC = sklearn.ensemble.GradientBoostingClassifier(n_estimators = 200)
    #GBC = svm.SVC(kernel = 'rbf',tol = 1e-3,class_weight = {1:5,0:1})
    #GBC = MLPClassifier(solver = 'adam',activation = 'tanh',alpha = 1,hidden_layer_sizes = (3),max_iter = 500,learning_rate = 'adaptive')
    GBC.fit(xt,yt,sample_weight=(5*yt+1*np.ones(len(yt))))
    #GBC.fit(xt,yt)
    y_pred = GBC.predict(x_test)
    yt_pred = GBC.predict(xt)
    
    lz = (list(zip(emp,y_pred)))
    if test==True:
        with open('base1.csv','w',newline='\n') as file:
            writer = csv.writer(file)
            writer.writerow(['id','left'])
            for l in lz:
                writer.writerow(l)
    if test ==False:
        #tn,fp,fn,tp = sklearn.metrics.confusion_matrix(y_test,y_pred).ravel()

        # Test accuracy
        num = 0
        denom = 0
        for i in range(len(y_test)):
            if y_test[i] ==1:
                if y_pred[i]==1:
                    num +=5
                denom +=5
            else:
                if y_pred[i]==0:
                    num +=1
                denom +=1
        print("Testing accuracy {}".format(num/denom))

        # Train accuracy
        num = 0
        denom = 0
        for i in range(len(yt)):
            if yt[i] ==1:
                if yt_pred[i]==1:
                    num +=5
                denom +=5
            else:
                if yt_pred[i]==0:
                    num +=1
                denom +=1
        #print("Training accuracy {}".format(num/denom))

def augment_data(xt,yt):
    l = len(yt)
    for i in range(l):
        if yt[i]==1:
            for i in range(4):
                xt = np.append(xt,np.reshape([xt[i,0],xt[i,1],xt[i,2],xt[i,3],xt[i,4],xt[i,5]],(1,6)),axis = 0)
                yt = np.append(yt,[yt[i]])
    return xt,yt

if __name__ == '__main__':
    compnames  = ['azalutpt', 'ejeyobsm', 'phcvroct', 'lgqwnfsg', 'wsmblohy', 'ydqdpmvi',
     'fqsozvpv','ocsicwng', 'oecfwdaq', 'oqvaqcak', 'nmxkgvmi', 'lydqevjo',
     'iqdwmigj','rcyiinms', 'pfmjacpm', 'ewpvmfbc', 'rcwkfavv', 'ujplihug',
     'rujnkvse','pkeebtfe', 'xccmgbjz', 'ojidyfnn', 'ugldwwzf', 'bucyzegb',
     'jnvpfmup','vcqsbirc', 'bhqczwkj', 'siexkzzo', 'fjslutlg', 'ylpksopb',
     'dmgwoqhz','bnivzbfi', 'jblrepyr', 'vwcdylha', 'yodaczsb', 'zptfoxyq','spfcrgea']
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--test',action='store_true',default=False)
    #args = parser.parse_args()

    test = False

    with open('train_data.p','rb') as fp:
        train_data = pickle.load(fp)
    with open('test_data.p','rb') as fp:
        test_data = pickle.load(fp)
    x_train,y_train = return_data(train_data)
    x_test,emp = return_data(test_data,True)

    if test == False:
        kf = KFold(n_splits = 5,shuffle = True, random_state = 42)
        for train_index, test_index in kf.split(x_train):
            xt = x_train[train_index]
            yt = y_train[train_index]

            #xt,yt = augment_data(xt,yt)
            x_test = x_train[test_index]
            y_test = y_train[test_index]
            main(xt,yt,x_test,y_test,emp,compnames,test = False)
    else:
        main(x_train,y_train,x_test,x_test,emp,compnames,test = True)

