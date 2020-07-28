# Use dataset with everything

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
from sklearn.utils import shuffle

# Hyperparameters:
n_estimators = 200 # For GBC
sample_weight_1 = 5 # Weight given to data point where employee has left the company
flag_augment = 0
default_avg_supp_remarks = 1
default_avg_supp_remarks_weighted = 0
default_n_supp_remarks = 0
default_avg_supported_ratio = 0.5
flag_supp_remarks_1 = 1 # Weigh the supported remarks by whether it was supported or opposed
flag_supp_remarks_2 = 1 # Divide the above by the total number of opinions given

# 0 to 0.1
c_supp_remarks = 0 # The constant in the expression of support that weighs the supported remark

# Use avg_remark_length or avg_remarks_length_weighted

flag_use_log_date = 0
default_avg_remarks_length = 88
default_avg_remarks_length_weighted = 88*17146
default_avg_ratio_support = 2
default_avg_remarks_support_ratings = 2
default_avg_remarks_support_ratings_last = 2

# 2.1 has better mean but 2.0 has better variance
default_avg_remarks_support_ratings_recent = 2.0

default_avg_remarks_support_ratings_weighted = 2
default_avg_remarks_support_ratings_weighted_relevant = 2
default_avg_remarks_support_ratings_weighted_recent = 2
c_remark_support = 0
default_avg_n_support = 2
default_total_remarks_length = 2
default_remark_supp_rating = 1
default_remark_support_rating_weighted = 1
default_emp_ratio_support = 0.5
default_emp_n_support = 0
default_emp_remark_support = [1]
flag_remark_supp_rating = 1
flag_remark_supp_rating_2 = 1

def funct(x,flag=0):
    #return np.log(x)
    if not flag:
        return x
    else:
        return np.exp(-x/np.mean(x))

def return_data(train_data,compnames,test=False):
    epoch = datetime.utcfromtimestamp(0)
    keys = train_data.keys()

    avg_ratings = [] # Average rating given
    n_ratings = [] # Number of ratings given
    avg_ratings_weighted = [] # Average rating given weighted by date on which it was given
    avg_ratings_log = [] # Average of log of ratings given
    std_ratings = [] # Standard deviation of ratings given
    avg_supp_remarks = [] # Average length of remarks supported
    avg_supp_remarks_weighted = [] # Average length of remarks opinionated weighted by date on which it was given
    n_supp_remarks = [] # Number of remarks supported
    comp = [] # Company
    avg_supported_ratio = [] 
    comp_onehot = [] # Company in one-hot form
    date = [] # Last rating date
    avg_remarks_length = [] # Average remark length
    n_remarks = [] # Number of remarks
    avg_remarks_length_weighted = []
    avg_ratio_support = [] # Average ratio of number of supports to total number of opinions given
    avg_n_support = [] # Average number of opinions given
    avg_remarks_support_ratings = [] # Average ratings of those who gave opinions on remarks
    avg_remarks_support_ratings_last = [] # Last rating of those who gave opinions on remarks
    avg_remarks_support_ratings_recent = []
    avg_remarks_support_ratings_weighted = [] # Average ratings of those who gave opinions on remarks weighted by date of remark
    avg_remarks_support_ratings_weighted_relevant = []
    avg_remarks_support_ratings_weighted_recent = []
 
    left = []
    total_remarks_length = []

    all_features = [avg_remarks_length_weighted,avg_remarks_support_ratings_weighted,avg_remarks_support_ratings_last,avg_remarks_support_ratings,avg_n_support,avg_ratio_support,avg_remarks_length,date,comp,n_supp_remarks,avg_ratings, n_ratings, avg_ratings_weighted, avg_ratings_log,std_ratings,avg_supp_remarks,avg_supp_remarks_weighted]
    ct0= 0
    ct = 0

    for k in keys:
        ct += 1
        """
        if ct<293:
            continue
        if ct>293:
            break
        print(ct)
        """

        emp_ratings = train_data[k]['rating']
        avg_ratings.append(np.mean(emp_ratings))
        n_ratings.append(len(emp_ratings))

        rating_dates = np.array(train_data[k]['rating_dates'])
        avg_ratings_weighted.append(np.mean(emp_ratings*funct(rating_dates)))
        avg_ratings_log.append(np.average(np.log(emp_ratings)))
        std_ratings.append(np.mean((emp_ratings*funct(rating_dates) - avg_ratings_weighted[-1])**2)**0.5)

        supp_remarks = np.array(train_data[k]['total_supp_remarks'][0])
        supports = np.array(train_data[k]['total_supp_remarks'][2])
        if supp_remarks.size==0: # If person hasn't given their opinion on any remark
            avg_supp_remarks.append(default_n_supp_remarks)
            avg_supp_remarks_weighted.append(default_avg_supp_remarks_weighted)
            n_supp_remarks.append(default_n_supp_remarks)
            avg_supported_ratio.append(default_avg_supported_ratio)
        else:
            n_supp_remarks.append(len(supp_remarks))
            avg_supp_remarks.append(np.mean(supp_remarks*(supports**flag_supp_remarks_1)/(n_supp_remarks[-1]**flag_supp_remarks_2)))
            emp_dates = []
            for j in train_data[k]['total_supp_remarks'][1]:
                emp_dates.append((j-epoch).days)
            avg_supp_remarks_weighted.append(np.mean(supp_remarks*emp_dates*((supports+c_supp_remarks)**flag_supp_remarks_1)/(n_supp_remarks[-1]**flag_supp_remarks_2)))
            avg_supported_ratio.append(np.sum(supports)/len(supports))

        comp.append(compnames.index(train_data[k]['comp']))
        company_name_onehot = np.zeros(len(compnames))
        company_name_onehot[compnames.index(train_data[k]['comp'])] = 1.0
        company_name_onehot = company_name_onehot[:-1]
        comp_onehot.append(company_name_onehot)

        date.append(funct((train_data[k]['lastdate'] - epoch).days,flag_use_log_date))

        if len(train_data[k]['remarks']) == 0: # If empty
            #average length is 89.14, so for those who dont have any comments, just putting average length
            avg_remarks_length.append(default_avg_remarks_length)
            avg_ratio_support.append(default_avg_ratio_support)
            avg_remarks_support_ratings.append(default_avg_remarks_support_ratings)
            avg_remarks_support_ratings_last.append(default_avg_remarks_support_ratings_last)
            avg_remarks_support_ratings_recent.append(default_avg_remarks_support_ratings_recent)
            avg_remarks_support_ratings_weighted.append(default_avg_remarks_support_ratings_weighted)
            avg_remarks_support_ratings_weighted_relevant.append(default_avg_remarks_support_ratings_weighted_relevant)
            avg_remarks_support_ratings_weighted_recent.append(default_avg_remarks_support_ratings_weighted_recent)
            avg_remarks_length_weighted.append(default_avg_remarks_length_weighted)
            avg_n_support.append(default_avg_n_support)
            total_remarks_length.append(default_total_remarks_length)
            ct0 += 1
        else:
            emp_avg_remark_length = []
            emp_avg_remark_length_weighted = []
            emp_ratio_support = []
            emp_n_support = []
            emp_avg_remark_support_ratings = []
            emp_avg_remark_support_ratings_last = []
            emp_avg_remark_support_ratings_recent = []
            emp_avg_remark_support_ratings_weighted = []
            emp_avg_remark_support_ratings_weighted_relevant = []
            emp_avg_remark_support_ratings_weighted_recent = []
            emp_remark_dates = []

            for j in train_data[k]['remarks'].keys():
                remark_data = train_data[k]['remarks'][j]
                emp_remark_lengths = remark_data[0]
                emp_remark_dates.append((remark_data[1]-epoch).days)
                emp_remark_support = remark_data[2]
                emp_remark_support_ratings = remark_data[3]

                emp_remark_support_ratings_recent = remark_data[4]
                a = np.where(emp_remark_support_ratings_recent==0)[0]
                if a.size != 0:
                    emp_remark_support_ratings_recent[a] = default_remark_support_rating                

                emp_remark_support_ratings_last = remark_data[5]
                a = np.where(emp_remark_support_ratings_last==0)[0]
                if a.size != 0:
                    emp_remark_support_ratings_last[a] = default_remark_support_rating

                emp_remark_support_ratings_weighted = remark_data[6]
                a = np.where(emp_remark_support_ratings_weighted==0)[0]
                if a.size != 0:
                    emp_remark_support_ratings_weighted[a] = default_remark_support_rating_weighted

                emp_remark_support_ratings_weighted_relevant = remark_data[7]
                a = np.where(emp_remark_support_ratings_weighted_relevant==0)[0]
                if a.size != 0:
                    emp_remark_support_ratings_weighted_relevant[a] = default_remark_support_rating_weighted

                emp_remark_support_ratings_weighted_recent = remark_data[8]
                a = np.where(emp_remark_support_ratings_weighted_recent==0)[0]
                if a.size != 0:
                    emp_remark_support_ratings_weighted_recent[a] = default_remark_support_rating_weighted

                emp_avg_remark_length_weighted.append(np.mean(emp_remark_lengths*(emp_remark_dates[-1])))
                if emp_remark_support.size == 0:
                    emp_ratio_support.append(default_emp_ratio_support)
                    emp_n_support.append(default_emp_n_support)
                    emp_remark_support = default_emp_remark_support
                else:
                    emp_ratio_support.append(np.sum(emp_remark_support)/len(emp_remark_support))
                    emp_n_support.append(np.sum(emp_remark_support))

                emp_avg_remark_length.append(np.mean(np.array(emp_remark_lengths)))
                emp_avg_remark_support_ratings.append(np.mean(emp_remark_support_ratings))
                emp_avg_remark_support_ratings_last.append(np.mean(emp_remark_support_ratings_last))
                emp_avg_remark_support_ratings_recent.append(np.mean(emp_remark_support_ratings_recent*(np.array(emp_remark_support)**flag_remark_supp_rating+c_remark_support)/(len(emp_remark_support)**flag_remark_supp_rating_2)))
                emp_avg_remark_support_ratings_weighted.append(np.mean(emp_remark_support_ratings_weighted))
                emp_avg_remark_support_ratings_weighted_relevant.append(np.mean(emp_remark_support_ratings_weighted_relevant*((np.array(emp_remark_support)**flag_remark_supp_rating+c_remark_support)/(len(emp_remark_support)**flag_remark_supp_rating_2))))
                emp_avg_remark_support_ratings_weighted_recent.append(np.mean(emp_remark_support_ratings_weighted_recent*((np.array(emp_remark_support)**flag_remark_supp_rating+c_remark_support)/(len(emp_remark_support)**flag_remark_supp_rating_2))))

            avg_remarks_length.append(np.mean(emp_avg_remark_length))
            n_remarks.append(len(emp_avg_remark_length))
            avg_remarks_length_weighted.append(np.mean(emp_avg_remark_length_weighted))
            avg_ratio_support.append(np.mean(emp_ratio_support))
            avg_n_support.append(np.mean(emp_n_support))
            avg_remarks_support_ratings.append(np.mean(emp_avg_remark_support_ratings))
            avg_remarks_support_ratings_last.append(np.mean(emp_avg_remark_support_ratings_last))
            avg_remarks_support_ratings_recent.append(np.mean(emp_avg_remark_support_ratings_recent))
            avg_remarks_support_ratings_weighted.append(np.mean(emp_avg_remark_support_ratings_weighted))
            avg_remarks_support_ratings_weighted_relevant.append(np.mean(emp_avg_remark_support_ratings_weighted_relevant))
            avg_remarks_support_ratings_weighted_recent.append(np.mean(emp_avg_remark_support_ratings_weighted_recent))

            total_remarks_length.append((avg_remarks_length[-1]*n_remarks[-1] + avg_supp_remarks[-1]*n_supp_remarks[-1])/(n_remarks[-1]+n_supp_remarks[-1]))
        if test==False:
            left.append(train_data[k]['left'])

    if test==False:
        left = np.array(left)

    x = (np.stack([comp,avg_ratio_support,date,avg_ratings_weighted,avg_remarks_length_weighted,avg_supp_remarks_weighted,avg_remarks_support_ratings_recent],-1))
    #x = np.append(x,np.reshape(comp_onehot,(x.shape[0],36)),axis = 1)
    if test== False:
        return x,left
    else:
        return x,keys

def main(xt,yt,x_test,y_test,emp,test=False):
    GBC = sklearn.ensemble.GradientBoostingClassifier(n_estimators = n_estimators,tol = 1e-6)
    #GBC = svm.SVC(kernel = 'rbf',tol = 1e-3,class_weight = {1:5,0:1})
    #GBC = MLPClassifier(solver = 'adam',activation = 'tanh',alpha = 1,hidden_layer_sizes = (3),max_iter = 500,learning_rate = 'adaptive')
    GBC.fit(xt,yt,sample_weight=(sample_weight_1*yt+1*np.ones(len(yt))))
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
        test_acc = num/denom
        #print("Testing accuracy {}".format(test_acc))

        # Train accuracy
        """
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
        print("Training accuracy {}".format(num/denom))
        """

        return test_acc

def augment_data(xt,yt,flag_augment):
    if flag_augment == 0:
        return xt,yt
    l = len(yt)
    check = 0
    for i in range(l):
        if yt[i]==1:
            if check == 1:
                check = 0
                continue
            for i in range(1):
                xt = np.append(xt,np.reshape([xt[i,0],xt[i,1],xt[i,2],xt[i,3],xt[i,4],xt[i,5],xt[i,6]],(1,7)),axis = 0)
                yt = np.append(yt,[yt[i]])
            check = 1
    return xt,yt

if __name__ == '__main__':
    compnames  = ['azalutpt', 'ejeyobsm', 'phcvroct', 'lgqwnfsg', 'wsmblohy', 'ydqdpmvi',
     'fqsozvpv','ocsicwng', 'oecfwdaq', 'oqvaqcak', 'nmxkgvmi', 'lydqevjo',
     'iqdwmigj','rcyiinms', 'pfmjacpm', 'ewpvmfbc', 'rcwkfavv', 'ujplihug',
     'rujnkvse','pkeebtfe', 'xccmgbjz', 'ojidyfnn', 'ugldwwzf', 'bucyzegb',
     'jnvpfmup','vcqsbirc', 'bhqczwkj', 'siexkzzo', 'fjslutlg', 'ylpksopb',
     'dmgwoqhz','bnivzbfi', 'jblrepyr', 'vwcdylha', 'yodaczsb', 'zptfoxyq','spfcrgea']

    test = True

    with open('train_data.p','rb') as fp:
        train_data = pickle.load(fp)
    with open('test_data.p','rb') as fp:
        test_data = pickle.load(fp)
    x_train,y_train = return_data(train_data,compnames)
    x_test,emp = return_data(test_data,compnames,True)
    """
    for i in range(len(y_train)):
        if y_train[i]==0:
            plt.plot(x_train[i,0],x_train[i,2],'ro')
        else:
            plt.plot(x_train[i,0],x_train[i,2],'go')
    plt.show()
    """
    if test == False:
        n_split = 5
        kf = KFold(n_splits = n_split,shuffle = True, random_state = 42)
        test_acc = []
        for train_index, test_index in kf.split(x_train):
            xt = x_train[train_index]
            yt = y_train[train_index]

            xt,yt = augment_data(xt,yt,flag_augment)
            x_test = x_train[test_index]
            y_test = y_train[test_index]
            test_acc.append(main(xt,yt,x_test,y_test,emp,test = False))
            #print(test_acc[-1])
        print("Mean Accuracy = {}, Std Dev = {}".format(np.mean(test_acc),np.std(test_acc)))
    else:
        main(x_train,y_train,x_test,x_test,emp,test = True)

