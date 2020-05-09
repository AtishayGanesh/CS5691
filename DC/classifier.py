import pickle
import numpy as np
import statistics
import sklearn
import sklearn.svm
import sklearn.ensemble

def main(compnames):
    with open('train_data.p','rb') as fp:
        train_data = pickle.load(fp)
    with open('test_data.p','rb') as fp:
        test_data = pickle.load(fp)
    keys = train_data.keys()
    ratings = []
    avg_remarks =[]
    avg_remarks_accept = []
    left = []
    comp = []
    ct0= 0
    for k in keys:
        ratings.append(statistics.mean(train_data[k]['rating']))
        company_name = np.zeros(len(compnames))
        company_name[compnames.index(train_data[k]['comp'])] = 1.0
        comp.append(company_name)
        if len(train_data[k]['remarks']) ==0:
            #average length is 89.14, so for those who dont have any comments, just putting average length
            avg_remarks.append(0)
            avg_remarks_accept.append(1)
            ct0 +=1
        else:
            avg_remarks_accept.append(statistics.mean([train_data[k]['remarks'][j][-1] for j in train_data[k]['remarks'].keys()]))

            avg_remarks.append(statistics.mean([train_data[k]['remarks'][j][0] for j in train_data[k]['remarks'].keys()]))

        left.append(train_data[k]['left'])
    left = np.array(left)
    ratings = (np.array(ratings)) 
    ratings = (ratings-np.average(ratings))/np.std(ratings)

    avg_remarks = np.array(avg_remarks)
    avg_remarks = (avg_remarks-np.average(avg_remarks))/np.std(avg_remarks)
    avg_remarks_accept = np.array(avg_remarks_accept)
    avg_remarks_accept = (avg_remarks_accept-np.average(avg_remarks_accept))/np.std(avg_remarks_accept)
    comp =np.array(comp)

    print(np.average(left))
    x = (np.stack([ratings,avg_remarks,avg_remarks_accept],-1))
    x = np.concatenate([x,comp],axis=-1)
    y = left
    xt = x[726:]
    yt = y[726:]
    x_test = x[0:726]
    y_test = y[0:726]
    GBC = sklearn.svm.SVC()
    print(x.shape,y.shape)
    GBC.fit(xt,yt,sample_weight=(4*left[726:]+np.ones(len(left[726:]))))
    y_pred=GBC.predict(x_test)
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

    main(compnames)    

