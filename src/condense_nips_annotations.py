import numpy as np
import os
import pandas as pd
import sys

# list file names and take their serial number
fnames = os.listdir(os.path.join(r"data\raw\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV","temporal_annotations_nips4b"))
csvs = [f[-7:] for f in fnames]

dfs = []
for c in csvs:
    try:
        # Read in each annotation file and concat to preceding dataframe.
        df = pd.read_csv(os.path.join(r"data\raw\NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV","temporal_annotations_nips4b","annotation_train"+c))
        new_df = (df.T.reset_index().T.reset_index(drop=True).set_axis([f'C{i+1}' for i in range(df.shape[1])], axis=1))
        new_df['IN FILE'] = "nips4b_birds_trainfile"+c
        new_df['SAMPLE RATE'] = 44100
        dfs.append(new_df)
    except:
        pass

nips_fix = pd.concat(dfs).reset_index(drop=True).rename(columns={'C1':'OFFSET','C2':'DURATION','C3':'MANUAL ID'})
nips_fix.to_csv(os.path.join(r"data\out","NIPS_Annotations_condensed.csv"))
nips_fix