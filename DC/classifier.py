import pickle
import numpy as np
import statistics
import sklearn
import sklearn.naive_bayes
import sklearn.ensemble
def main():
    with open('train_data.p','rb') as fp:
        train_data = pickle.load(fp)
    with open('test_data.p','rb') as fp:
        test_data = pickle.load(fp)
    keys = train_data.keys()
    ratings = []
    avg_remarks =[]
    left = []
    ct0= 0
    for k in keys:
        ratings.append(statistics.mean(train_data[k]['rating']))
        if len(train_data[k]['remarks']) ==0:
            #average length is 89.14, so for those who dont have any comments, just putting average length
            avg_remarks.append(0)
            ct0 +=1
        else:
            avg_remarks.append(statistics.mean([train_data[k]['remarks'][j][0] for j in train_data[k]['remarks'].keys()]))

        left.append(train_data[k]['left'])
    left = np.array(left)
    ratings = (np.array(ratings)) 
    ratings = (ratings-np.average(ratings))/np.std(ratings)

    avg_remarks = np.array(avg_remarks)
    avg_remarks = (avg_remarks-np.average(avg_remarks))/np.std(avg_remarks)
    print(np.average(left))
    print(np.average(avg_remarks))
    print(np.average(ratings))
    x = (np.stack([ratings,avg_remarks],-1))
    y = left
    xt = x[0:2800]
    yt = y[0:2800]
    x_test = x[2800:]
    y_test = y[2800:]
    GBC = sklearn.ensemble.RandomForestClassifier()
    print(x.shape,y.shape)
    GBC.fit(xt,yt,sample_weight=(4*left[0:2800]+np.ones(len(left[0:2800]))))
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
    main()    

