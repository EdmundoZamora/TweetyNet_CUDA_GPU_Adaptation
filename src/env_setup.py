import os
import json
import shutil

basedir = os.path.dirname(__file__)

def make_datadir():
    data_loc = os.path.join(basedir, '..', 'data')
    
    if os.path.exists(data_loc):
       shutil.rmtree(data_loc,ignore_errors=True)

    for d in ['raw', 'temp', 'out']: #'out/separate_evaluations' try this one better
        os.makedirs(os.path.join(data_loc, d), exist_ok=True)

    # os.makedirs(r"data\out\separate_evaluations")
    return

#make_datadir() #run this file to reset the data directory