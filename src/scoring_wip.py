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

evals = pd.read_csv(os.path.join("data/out","Evaluation_on_data.csv")) #r"data\out\Evaluation_on_data.csv") 
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
file_filt = evals[evals['file'] == curr_file]
file_filt.to_csv(os.path.join("data/out","nips4b_birds_classificationfile001.csv"))
#endregion