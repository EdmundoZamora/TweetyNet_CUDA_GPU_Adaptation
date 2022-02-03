import pandas as pd
import numpy as np
import os
import re
from tabulate import tabulate

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

evals = pd.read_csv(r"data\out\Evaluation_on_data.csv") #os.path.join("data\out","Evaluation_on_data.csv")) #
dc = evals.copy(deep=True)
wav = evals['file'].drop_duplicates() 
wav.index = dc['file'].drop_duplicates().str[-7:-4].values
wav = wav.sort_index(ascending = True).values

#region
# curr_file = wav[8] 
# file_filt = evals[evals['file'] == curr_file]
# print(tabulate(file_filt, headers='keys', tablefmt='psql')) 
#endregion

for i in range(1):
    curr_file = wav[i]
    file_filt = evals[evals['file'] == curr_file]
    morfi = "annotation_train"+curr_file[-7:-4]+".csv"
    print(morfi)
    try:
        real = pd.read_csv(os.path.join("data/raw/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/temporal_annotations_nips4b",morfi))
        print(tabulate(real, headers='keys', tablefmt='psql'))
    except:
        continue


#region
curr_file = wav[0] 
evals['acc'] = (evals['pred']==evals['label']).astype(int)

evals['cfnmtx'] = ''
evals.loc[(evals['pred'] == 0) & (evals['label'] == 1), 'cfnmtx'] = 'FN'
evals.loc[(evals['pred'] == 0) & (evals['label'] == 0), 'cfnmtx'] = 'TN'
evals.loc[(evals['pred'] == 1) & (evals['label'] == 0), 'cfnmtx'] = 'FP'
evals.loc[(evals['pred'] == 1) & (evals['label'] == 1), 'cfnmtx'] = 'TP'

file_filt = evals[evals['file'] == curr_file]
file_filt.to_csv(os.path.join("data/out","nips4b_birds_classificationfile001.csv"))
#endregion

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        elif y_actual[i]==y_hat[i]==0:
           TN += 1
        else: # y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
    """
    Given a dataframe with at least
    two categorical columns, create a 
    confusion matrix of the count of the columns
    cross-counts
    
    use like:
    
    >>> confusion_matrix(test_df, 'actual_label', 'predicted_label')
    """
    return (
            df
            .groupby([col1, col2])
            .size()
            .unstack(fill_value=0)
            )

print('\n')
print('---------------------------------------------------------------------')
print('\n')
print(perf_measure(file_filt['pred'].values, file_filt['label'].values))
print('\n')
print('correct')
print('\n')
print(confusion_matrix(file_filt,'pred','label'))
print('\n')
print(file_filt['cfnmtx'].value_counts())
print('\n')
print('---------------------------------------------------------------------')