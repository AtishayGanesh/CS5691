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

def return_data(train_data,test=False):
    epoch = datetime.utcfromtimestamp(0)
    keys = train_data.keys()
    ratings = []
    ratings2 = []
    avg_remarks =[]
    avg_remarks_accept = []
    std_remarks =[]
    std_ratings =[]
    left = []
    comp = []
    date = []
    ct0= 0

    for k in keys:
        l = train_data[k]['rating']

        ratings.append((statistics.mean(l)))
        ratings2.append(np.average(np.log(l)))

        std_ratings.append(stats.skew(l))
        company_name = np.zeros(len(compnames))
        company_name[compnames.index(train_data[k]['comp'])] = 1.0
        #comp.append(company_name)
        comp.append(compnames.index(train_data[k]['comp']))
        date.append((train_data[k]['lastdate'] - epoch).total_seconds())
        if len(train_data[k]['remarks']) ==0:
            #average length is 89.14, so for those who dont have any comments, just putting average length
            avg_remarks.append(89.14)
            avg_remarks_accept.append(1)
            std_remarks.append(1)
            ct0 +=1
        else:
            avg_remarks_accept.append(statistics.mean([train_data[k]['remarks'][j][-1] for j in train_data[k]['remarks'].keys()]))
            std_remarks.append((statistics.mean([np.abs(np.log(1+train_data[k]['remarks'][j][-1])) for j in train_data[k]['remarks'].keys()])))
            avg_remarks.append(statistics.mean([train_data[k]['remarks'][j][0] for j in train_data[k]['remarks'].keys()]))
        if test==False:
            left.append(train_data[k]['left'])
    if test==False:
        left = np.array(left)
    ratings = (np.array(ratings)) 
    ratings = (ratings-np.average(ratings))/np.std(ratings)
    ratings2 = (np.array(ratings2)) 
    ratings2 = (ratings2-np.average(ratings2))/np.std(ratings2)

    std_ratings = np.array(ratings)
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
    x = (np.stack([ratings,ratings2,std_ratings,avg_remarks,avg_remarks_accept,std_remarks,date,comp],-1))
    #x = np.concatenate([x,comp],axis=-1)
    if test== False:
        return x,left
    else:
        return x,keys

def main(compnames,test=False):
    with open('train_data.p','rb') as fp:
        train_data = pickle.load(fp)
    with open('test_data.p','rb') as fp:
        test_data = pickle.load(fp)
    xt,yt = return_data(train_data)
    x_test,emp = return_data(test_data,True)

    if test ==False:
        x_test = xt[2600:]
        y_test = yt[2600:]
        xt = xt[0:2600]
        yt = yt[0:2600]


    GBC = sklearn.ensemble.GradientBoostingClassifier()
    print(xt.shape,yt.shape)
    GBC.fit(xt,yt,sample_weight=(4*yt+np.ones(len(yt))))
    y_pred=GBC.predict(x_test)
    print(y_pred)
    lz = (list(zip(emp,y_pred)))
    if test==True:
        with open('base1.csv','w',newline='\n') as file:
            writer = csv.writer(file)
            writer.writerow(['id','left'])
            for l in lz:
                writer.writerow(l)
    if test ==False:
        print(np.average(y_pred),np.average(y_test))
        print(sklearn.metrics.confusion_matrix(y_test,y_pred))
        tn,fp,fn,tp = sklearn.metrics.confusion_matrix(y_test,y_pred).ravel()
        print(tn)
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
        print(num/denom)










if __name__ == '__main__':
    compnames  = ['azalutpt', 'ejeyobsm', 'phcvroct', 'lgqwnfsg', 'wsmblohy', 'ydqdpmvi',
     'fqsozvpv','ocsicwng', 'oecfwdaq', 'oqvaqcak', 'nmxkgvmi', 'lydqevjo',
     'iqdwmigj','rcyiinms', 'pfmjacpm', 'ewpvmfbc', 'rcwkfavv', 'ujplihug',
     'rujnkvse','pkeebtfe', 'xccmgbjz', 'ojidyfnn', 'ugldwwzf', 'bucyzegb',
     'jnvpfmup','vcqsbirc', 'bhqczwkj', 'siexkzzo', 'fjslutlg', 'ylpksopb',
     'dmgwoqhz','bnivzbfi', 'jblrepyr', 'vwcdylha', 'yodaczsb', 'zptfoxyq','spfcrgea']
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',action='store_true',default=False)
    args = parser.parse_args()
    main(compnames,test = args.test)
