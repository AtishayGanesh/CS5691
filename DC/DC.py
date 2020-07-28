"""
Pickle File has:

comp: company, scalar
lastdate: last rating data, scalar
emp: emp id, scalar
rating: array of all ratings
rating_dates: array of all dates of ratings
left: (only for training data) if the employee has left the company
remarks: dictionary keyed by rid where each element is a list with the following: (if empty will be an empty dict) (default values are all [0])
    txt: array of length of remarks made by the employee (element = 0 if there was no remark)
    remarkdate: the date when the remark was made (element = utcfromtimestamp(0) if no date was given)
    support: list of supports/oppositions given by employees
    ratings_support: list average ratings of people who have given support/oppositions (=[0] if doesn't exist)
    ratings_support_recent: list of the most recent rating given by all the employees who gave an opinion on the remark
    ratings_support_last: list of last rating given by each person who gave support/opposition (=[0] if doesn't exist)
    ratings_support_weighted: list of average of supporting ratings of the person weighted by date of rating (=[0] if doesn't exist)
    ratings_support_weighted_relevant: same as previous but only considering the ratings before the remark
    ratings_support_weighted_recent: using only the closest rating to the remark
total_supp_remarks: list of the following:
    total_supp_remarks: list of length of remarks the employee has supported (element = 0 if no remark or empty remark)
    remark_date: list of the dates on which the remark was given
    support: list of whether the employee supported/opposed
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import sys

def organize_data(data,ratings,remarks,remarks_supp_opp,test):
    ids = data['id'].to_numpy()
    dict_info = {}
    ct = 0
    for e_id in ids:
        ct+=1
        """
        if ct<2:
            continue
        if ct>2:
            break
        """
        print('at ct: ',ct,'out of ',len(ids))

        # Basic employee data
        p_info = {}
        loc = data.loc[data['id'] == e_id]
        comp = list(loc['comp'])[0]
        p_info['comp'] = comp
        d  = list(loc['lastratingdate'])[0]
        p_info['lastdate'] = datetime.strptime(d,"%d-%m-%Y")
        emp  = list(loc['emp'])[0]
        p_info['emp'] = int(emp)

        if test == False:
            leave  = list(loc['left'])[0]
            p_info['left'] = int(leave)

        # Employee rating data
        loc_ratings = ratings.loc[(ratings['emp']==emp)&(ratings['comp'] ==comp)]
        dates = [(datetime.strptime(xx,"%d-%m-%Y")-datetime.utcfromtimestamp(0)).days for xx in list(loc_ratings['Date'])]
        lr = loc_ratings['rating'].to_numpy()
        p_info['rating'] = lr
        p_info['rating_dates'] = dates

        # Employee remark data
        loc_remarks = remarks.loc[(remarks['emp']==emp)&(remarks['comp']==comp)]
        if loc_remarks.empty == False:

            ## Basic remark data
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

            ## Data of opinions given on remarks
            date_cnt = -1
            for rid in remark_ids:
                date_cnt+=1
                if len(remark_dict[rid])==2: # If not, we are discarding the remark
                    loc_rso =remarks_supp_opp.loc[remarks_supp_opp['remarkId']==rid]
                    empids = list(loc_rso['emp'])
                    comps = list(loc_rso['comp'])
                    support = list(loc_rso['support'])

                    support = np.array(support)
                    
                    ratings_support = []
                    ratings_support_last = []
                    ratings_support_recent = []
                    ratings_support_weighted = []
                    ratings_support_weighted_relevant = []
                    ratings_support_weighted_recent = []
                    #remark_date = datetime.strptime(list(remarks.loc[(remarks['remarkId']==rid)&(remarks['emp']==emp)]['remarkDate'])[0],"%d-%m-%Y")
                    remark_day = (remarkDate[date_cnt] - datetime.utcfromtimestamp(0)).days
                    e = 0
                    for x in empids:
                        if x<0: # Discard such values
                            continue
                        company_x = comps[e] # Just to speed up the search
                        list_of_ratings = np.array(list(ratings.loc[(ratings['emp']==x)&(ratings['comp']==company_x)]['rating']))
                        ratings_dates = np.array([(datetime.strptime(xx,"%d-%m-%Y")-datetime.utcfromtimestamp(0)).days for xx in list(ratings.loc[(ratings['emp']==x)&(ratings['comp']==company_x)]['Date'])])
                        sorted_indexes = np.argsort(ratings_dates)
                        ratings_dates = ratings_dates[sorted_indexes]
                        list_of_ratings = list_of_ratings[sorted_indexes]
                        ratings_support.append(np.mean(list_of_ratings)) # Average rating given by each person who gave an opinion on the remark
                        ratings_support_last.append(list_of_ratings[-1]) # Last rating given by each person who gave an opinion on the remark
                        earlier_dates = np.where(ratings_dates<=remark_day)[0]
                        if earlier_dates.size == 0:
                            earlier_dates = [0]
                        ratings_support_recent.append(list_of_ratings[earlier_dates[-1]]) # Most recent rating given by the person
                        ratings_support_weighted.append(np.mean(list_of_ratings*ratings_dates)) # Weighted mean of all ratings
                        ratings_support_weighted_relevant.append(np.mean(np.mean((list_of_ratings*ratings_dates)[earlier_dates]))) # Weighted mean of only those ratings before the remark
                        ratings_support_weighted_recent.append(list_of_ratings[earlier_dates[0]]*ratings_dates[earlier_dates[0]]) # Closest rating to the remark weighted by the date of rating
                        e+=1
                    if ratings_support == []:
                        ratings_support = [0]
                        ratings_support_last = [0]
                        ratings_support_recent = [0]
                        ratings_support_weighted = [0]
                        ratings_support_weighted_relevant = [0]
                        ratings_support_weighted_recent = [0]
                    remark_dict[rid] = list(remark_dict[rid])+[support,ratings_support,ratings_support_recent,ratings_support_last,ratings_support_weighted,ratings_support_weighted_relevant,ratings_support_weighted_recent]
        else:
            remark_dict = {}
        p_info['remarks'] = remark_dict
        
        # Data on what an employee supported/opposed
        loc_support = remarks_supp_opp.loc[(remarks_supp_opp['emp']==emp)&(remarks_supp_opp['comp']==comp)]
        remarkids = list(loc_support['remarkId'])
        support = np.array(list(loc_support['support']))
        total_supp_remarks = []
        remark_date = []
        for rid in remarkids:
            loc_remarks = remarks.loc[(remarks['remarkId']==rid)]
            remark_len = list(loc_remarks['txt'])
            if remark_len == []: # Remark doesn't exist
                    total_supp_remarks.append(0)
                    remark_date.append(datetime.utcfromtimestamp(0))
            else:
                i = remark_len[0]
                remark_date.append(datetime.strptime(list(remarks.loc[(remarks['remarkId']==rid)]['remarkDate'])[0],"%d-%m-%Y"))
                if i!=i: # Might be an empty remark
                    total_supp_remarks.append(0)
                else:
                    total_supp_remarks.append(len(i))
        try:
            temp = total_supp_remarks*support
            total_supp_remarks = np.array(total_supp_remarks)
        except:
            print(ct,len(total_supp_remarks),len(support),len(remarkids))
            print("ER")
            break
        p_info['total_supp_remarks'] = [total_supp_remarks,remark_date,support]
        
        dict_info[int(e_id)] = p_info

    if test == False:
        file = 'train_data.p'
    else:
        file = 'test_data.p'
    with open(file, 'wb') as fp:
        pickle.dump(dict_info, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    # These files have duplicate entries removed and also rows with emp<0 were removed
    train = pd.read_csv('train_uniq.csv')
    test = pd.read_csv('test.csv')
    ratings = pd.read_csv('ratings_uniq.csv')
    remarks = pd.read_csv('remarks_uniq_edit.csv')
    remarks_supp_opp = pd.read_csv('remarks_supp_opp_uniq_edit.csv',low_memory=False)

    organize_data(train,ratings,remarks,remarks_supp_opp,test = False) # Organise the training data
    organize_data(test,ratings,remarks,remarks_supp_opp,test = True) # Organise the test data