import numpy as np
import os
import pandas as pd
import sys
import tabulate 
from tabulate import tabulate



# "C:\Users\lianl\Repositories\TweetyNet_CUDA_GPU_Adaptation\data\raw\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV\temporal_annotations_nips4b\annotation_train001.csv"

# tests

# list files and print file names in list

# print a dataframe

# save dataframe to an out dir
# concat to out dir
# open the out dir df


# lists all the files in this dir
# for loop
# read each csv as a pandas dataframe
# concat each dataframe, to output dataframe in out dir
# 


# def find_tags(data_path, folder):
#     fnames = os.listdir(os.path.join(data_path, "temporal_annotations_nips4b"))
#     csvs = []
#     for f in fnames:
#         #print(f)
#         csvs.append(pd.read_csv(os.path.join(data_path, "temporal_annotations_nips4b", f), index_col=False, names=["start", "duration", "tag"]))
#         # instead of doing this we just get that info from one dataframe.! WIP
#     # print(type(csvs[0]))
#     # return
#     return csvs

fnames = os.listdir(os.path.join('data/raw/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', "temporal_annotations_nips4b"))

# fnames = os.listdir(r"C:\Users\lianl\Repositories\TweetyNet_CUDA_GPU_Adaptation\data\raw\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV\temporal_annotations_nips4b")
csvs = []
for f in fnames:
    # print(f)
    # print(f[-7:])
    csvs.append(pd.read_csv(os.path.join(r'data\raw\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV', "temporal_annotations_nips4b","annotation_train"+f[-7:])))

# print(fnames)
# print(csvs)