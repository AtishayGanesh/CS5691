import pandas as pd
import numpy as np
from datetime import datetime
import pickle

def organize_train_data(train,ratings,remarks,remarks_supp_opp,smooth=0.1):
    ids = train['id'].to_numpy()
    dict_info = {}
    ct = 0
    for id in ids:
        ct +=1
        print('at ct: ',ct,'out of ',len(ids))
        p_info = {}
        loc = train.loc[train['id'] == id]
        comp = list(loc['comp'])[0]
        p_info['comp']=comp
        d  = list(loc['lastratingdate'])[0]
        p_info['lastdate'] = datetime.strptime(d,"%d-%m-%Y")
        emp  = list(loc['emp'])[0]
        p_info['emp'] = int(emp)
        leave  = list(loc['left'])[0]
        p_info['left'] = int(leave)

        loc_ratings = ratings.loc[(ratings['emp']==emp)&(ratings['comp'] ==comp)]
        lr = loc_ratings['rating'].to_numpy()
        lr_avg = lr[0]
        for i in lr:
            lr_avg = smooth*(i)+(1-smooth)*lr_avg
        p_info['rating'] = lr
        p_info['avgrating'] = lr_avg

        loc_remarks = remarks.loc[(remarks['emp']==emp)&(remarks['comp'] ==comp)]
        if loc_remarks.empty == False:
            remark_ids = list(loc_remarks['remarkId'])
            txt = []
            for i in list(loc_remarks['txt']):
                if i !=i:
                    txt.append(0)
                else:
                    txt.append(len(i))
            remarkDate = []
            for i in list(loc_remarks['remarkDate']):
                if i!=i:
                    i1 = "01-01-1970"
                    remarkDate.append(datetime.strptime(i1,"%d-%m-%Y"))

                else:
                    remarkDate.append(datetime.strptime(i,"%d-%m-%Y"))
            remark_dict = dict(zip(remark_ids,list(zip(txt,remarkDate))))
            for rid in remark_ids:
                if len(remark_dict[rid])==2:
                    loc_rso =remarks_supp_opp.loc[remarks_supp_opp['remarkId']==rid]
                    empids = list(loc_rso['emp'])
                    support = list(loc_rso['support'])
                    if len(support) ==0:
                        
                        ratio_true = 0.5
                    else:
                        ratio_true = sum(support)/len(support)

                    remark_dict[rid] = list(remark_dict[rid])+[list(map(list,zip(empids,support))),ratio_true]

        else:
            remark_dict = {}

        p_info['remarks'] = remark_dict

        dict_info[int(id)] = p_info
        #raise AssertionError
    with open('train_data.p', 'wb') as fp:
        pickle.dump(dict_info, fp, protocol=pickle.HIGHEST_PROTOCOL)




def organize_test_data(test,ratings,remarks,remarks_supp_opp,smooth=0.1):
    ids = test['id'].to_numpy()
    dict_info = {}
    ct = 0

    for id in ids:
        ct +=1
        print('at ct: ',ct,'out of ',len(ids))
        p_info = {}
        loc = test.loc[test['id'] == id]
        comp = list(loc['comp'])[0]
        p_info['comp']=comp
        d  = list(loc['lastratingdate'])[0]
        p_info['lastdate'] = datetime.strptime(d,"%d-%m-%Y")
        emp  = list(loc['emp'])[0]
        p_info['emp'] = int(emp)

        loc_ratings = ratings.loc[(ratings['emp']==emp)&(ratings['comp'] ==comp)]
        lr = loc_ratings['rating'].to_numpy()
        lr_avg = lr[0]
        for i in lr:
            lr_avg = smooth*(i)+(1-smooth)*lr_avg
        p_info['rating'] = lr
        p_info['avgrating'] = lr_avg

        loc_remarks = remarks.loc[(remarks['emp']==emp)&(remarks['comp'] ==comp)]
        if loc_remarks.empty == False:
            remark_ids = list(loc_remarks['remarkId'])
            txt = []
            for i in list(loc_remarks['txt']):
                if i !=i:
                    txt.append(0)
                else:
                    txt.append(len(i))
            #remarkDate = [datetime.strptime(i,"%d-%m-%Y") for i in list(loc_remarks['remarkDate'])]
            remarkDate = []
            for i in list(loc_remarks['remarkDate']):
                if i!=i:
                    i1 = "01-01-1970"
                    remarkDate.append(datetime.strptime(i1,"%d-%m-%Y"))

                else:
                    remarkDate.append(datetime.strptime(i,"%d-%m-%Y"))
            remark_dict = dict(zip(remark_ids,list(zip(txt,remarkDate))))

            for rid in remark_ids:
                if len(remark_dict[rid])==2:
                    loc_rso =remarks_supp_opp.loc[remarks_supp_opp['remarkId']==rid]
                    empids = list(loc_rso['emp'])
                    support = list(loc_rso['support'])
                    if len(support) ==0:
                        
                        ratio_true = 0.5
                    else:
                        ratio_true = sum(support)/len(support)

                    remark_dict[rid] = list(remark_dict[rid])+[list(map(list,zip(empids,support))),ratio_true]

        else:
            remark_dict = {}

        p_info['remarks'] = remark_dict

        dict_info[int(id)] = p_info
    with open('test_data.p', 'wb') as fp:
        pickle.dump(dict_info, fp, protocol=pickle.HIGHEST_PROTOCOL)











if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    ratings = pd.read_csv('ratings.csv')
    remarks = pd.read_csv('remarks.csv')
    remarks_supp_opp = pd.read_csv('remarks_supp_opp.csv',low_memory=False)

    #organize_train_data(train,ratings,remarks,remarks_supp_opp)

    organize_test_data(test,ratings,remarks,remarks_supp_opp)

    # print(train)
    # print(len(train['id'].unique()))
    # print(len(train['comp'].unique()))
    # print(remarks)
    # print(len(remarks['emp'].unique()))

