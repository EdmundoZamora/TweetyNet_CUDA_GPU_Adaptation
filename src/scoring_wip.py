import pandas as pd
import numpy as np
import os
import sys
import re
from tabulate import tabulate

import pandas as pd
import os
import matplotlib.pyplot as pl

import seaborn as sns
import matplotlib.pyplot as plt

'''
Scoring Precision and Recall is currently a work in progress for Q2
'''
# region
# evals = pd.read_csv(os.path.join("data/out","Evaluation_on_data.csv"))
#print(tabulate(evals, headers='keys', tablefmt='psql'))
#print(evals.shape)

#time = pd.read_csv(os.path.join("data/out","Time_intervals.csv"))
#print(tabulate(time, headers='keys', tablefmt='psql'))
#print(time.shape)

# print(f"91800/216 = {float(91800/216)}")

# nu_time = pd.concat([time]*425, ignore_index=True)
# #print(tabulate(nu_time, headers='keys', tablefmt='psql'))
# print(nu_time.shape)

# extracted_col = nu_time["start_time"]
# evals_timed = evals.join(extracted_col)
# evals_timed.to_csv(os.path.join("data/out","Time_interval_Evaluations.csv"))
#print(tabulate(evals_timed, headers='keys', tablefmt='psql'))
#print(evals_timed.shape)
# add three new columns, two for start and end time intervals, and one for trainfile id

# use trainfile id to pull annotation csv
# compare annotation time intervals with the binary classifications. 
# dont filter just overlay and compare
# analyse the difference, are the evaluations, spot on? within true? encapsulate true?

# using the real nips4b annotations to check if tweetynet detected in that time interval

# iteratively filter for only the trainfile in interest. 
# compare and score, add to a score chart (list)

# next trainfile

# def atof(text):
#     try:
#         retval = float(text)
#     except ValueError:
#         retval = text
#     return retval

# def natural_keys(text):
#     '''
#     alist.sort(key=natural_keys) sorts in human order
#     http://nedbatchelder.com/blog/200712/human_sorting.html
#     (See Toothy's implementation in the comments)
#     float regex comes from https://stackoverflow.com/a/12643073/190597
#     '''
#     return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
# endregion

'''
~ Gets & sorts unique file names 
-> searches for its corresponding annotations 
-> evaluate distance and duration of each real start time to classified frame start times
-> measure presicion and recall
'''
def file_score(num_files):
        
    evals = pd.read_csv(os.path.join("data","out","Evaluation_on_data.csv")) #os.path.join("data\out","Evaluation_on_data.csv")) #

    os.makedirs(os.path.join("data","out","separate_evaluations"))

    dc = evals.copy(deep=True)
    wav = evals['file'].drop_duplicates() 
    wav.index = dc['file'].drop_duplicates().str[-7:-4].values # last three numbers
    wav = wav.sort_index(ascending = True).values
    
    # print(wav)
    #region
    # curr_file = wav[8] 
    # file_filt = evals[evals['file'] == curr_file]
    # print(tabulate(file_filt, headers='keys', tablefmt='psql')) 
    #endregion
    frame = pd.DataFrame()
    file_names = []
    rates_dicts = []
    # similar_scores = []

    for i in range(num_files): # num files? or number on file?
        
        curr_file = wav[i]

        file_filt = evals[evals['file'] == curr_file].copy(deep = True)

        file_filt['Acc'] = (file_filt['pred']==file_filt['label']).astype(int)
        file_filt['cfnmtx'] = ''
        file_filt.loc[(file_filt['pred'] == 0) & (file_filt['label'] == 1), 'cfnmtx'] = 'FN'
        file_filt.loc[(file_filt['pred'] == 0) & (file_filt['label'] == 0), 'cfnmtx'] = 'TN'
        file_filt.loc[(file_filt['pred'] == 1) & (file_filt['label'] == 0), 'cfnmtx'] = 'FP'
        file_filt.loc[(file_filt['pred'] == 1) & (file_filt['label'] == 1), 'cfnmtx'] = 'TP'

        # print(tabulate(file_filt, headers='keys', tablefmt='psql'))
        # break

        morfi = "annotation_train"+curr_file[-7:-4]+".csv"
        
        # print(acc_score)
        
        # print(morfi)
        try:
            # real = pd.read_csv(os.path.join("data/raw/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/temporal_annotations_nips4b",morfi))
            # print(tabulate(real, headers='keys', tablefmt='psql'))
            file_filt.to_csv(os.path.join("data/out/separate_evaluations","nips4b_birds_classificationfile"+curr_file[-7:-4]+".csv"))
            # acc_score = file_filt['Acc'].to_list()

            # print('\n')
            
            # print('\n')
            # print('---------------------------------------------------------------------')
            # print('\n')
            # print(f'{morfi} Rates and Accuracy')
            # print('\n')
            # print(confusion_matrix(file_filt,'pred','label'))
            # rates = file_filt.groupby(['pred','label']).size().unstack(fill_value=0)
            # print(rates)
            # print('\n')
            
            deekt = dict(file_filt['cfnmtx'].value_counts())

            # act = sum(acc_score)/len(acc_score)

            file_names.append(morfi)
            rates_dicts.append(deekt)
            # similar_scores.append(act)
        except:
            pass

        # print(deekt)
        # print(act)
        # print('\n')
        # print('---------------------------------------------------------------------')
        

    d = {"FILE": file_names,"RATES_COUNT": rates_dicts} # ,"SIMILARITY":similar_scores data_length and number of annotations
        # print(d)
    scores = pd.DataFrame(d)

    scores = pd.concat([scores.drop(['RATES_COUNT'], axis=1), scores['RATES_COUNT'].apply(pd.Series)], axis=1)
    frame = frame.append(scores)

    frame = frame.fillna(0)

    frame['ERROR_RATE'] = frame.apply (lambda row: error_rate(row), axis=1)
    frame['ACCURACY'] = frame.apply (lambda row: accuracy(row), axis=1)
    frame['PRECISION'] = frame.apply (lambda row: precision(row), axis=1)
    # frame['RECALL'] = frame.apply (lambda row: sensitivity(row), axis=1)
    # frame['TRUE_NEGATIVE_RATE'] = frame.apply (lambda row: specificity(row), axis=1)
    # frame['FALSE_POSITIVE_RATE'] = frame.apply (lambda row: false_pos_rate(row), axis=1)
    # frame['F1_SCORE'] = frame.apply (lambda row: false_pos_rate(row), axis=1)
    
    frame.to_csv(os.path.join("data/out","Concluding_model_metrics.csv"))
#region
# def perf_measure(y_actual, y_hat):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0

#     for i in range(len(y_hat)): 
#         if y_actual[i]==y_hat[i]==1:
#            TP += 1
#         elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
#            FP += 1
#         elif y_actual[i]==y_hat[i]==0:
#            TN += 1
#         else: # y_hat[i]==0 and y_actual[i]!=y_hat[i]:
#            FN += 1

#     return(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

# def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
#     """
#     Given a dataframe with at least
#     two categorical columns, create a 
#     confusion matrix of the count of the columns
#     cross-counts
    
#     use like:
    
#     >>> confusion_matrix(test_df, 'actual_label', 'predicted_label')
#     """
#     return (
#             df
#             .groupby([col1, col2])
#             .size()
#             .unstack(fill_value=0)
#             )

# print('\n')
# print('---------------------------------------------------------------------')
# print('\n')
# print(perf_measure(file_filt['pred'].values, file_filt['label'].values))
# print('\n')
# print('correct')
# print('\n')
# print('---------------------------------------------------------------------')
# print('\n')
# print(confusion_matrix(file_filt,'pred','label'))
# print('\n')
# print(file_filt['cfnmtx'].value_counts())
# print('\n')
# print('---------------------------------------------------------------------')
#endregion

def error_rate(row):
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    error = (FP + FN)/(TP + TN + FN + FP)
    return error

def accuracy(row):
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    # return (TP,TN,FP,FN)
    accura = (TP + TN)/(TP + TN + FN + FP) 
    return accura

def sensitivity(row): # Recall or True positive rate
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    sense = (TP)/(TP+FN)
    return sense

def specificity(row): # True negative rate
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    specific = (TN)/(TN + FP)
    return specific

def precision(row): # positive predictive value
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    if TP == 0 and FP == 0:
        return 0
    else:
        prec = (TP)/(TP+FP)
        return prec

def false_pos_rate(row): 
    TP,TN,FP,FN = row['TP'],row['TN'],row['FP'],row['FN']
    fpr = (FP)/(TN+FP)
    return fpr

def f1_score(row):
    prec,rec = row['PRECISION'],row['RECALL']
    fpr = (2*prec*rec)/float(prec+rec)
    return fpr

# file_score(2)
