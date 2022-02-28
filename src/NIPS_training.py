import os
import sys
import csv
import math
import pickle
import shutil
from collections import Counter
from datetime import datetime

from scoring_wip import*
from graphs import*
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
# from torchsummary import summary

# region
from torch import cuda
import torch
from torch import LongTensor
from torch import nn
from torch.utils.data import DataLoader
from network import TweetyNet
import librosa
from librosa import display
import scipy.signal as scipy_signal
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from glob import glob
#endregion

#from microfaune.audio import wav2spc, create_spec, load_wav
from TweetyNetAudio import wav2spc, create_spec, load_wav
import random
from CustomAudioDataset import CustomAudioDataset
from TweetyNetModel import TweetyNetModel

from Load_data_functions import load_dataset, load_pyrenote_dataset


def apply_features(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found):
    
    print("----------------------------------------------------------------------------------------------")
    print("\n")
    print("IGNORE MISSING WAV FILES - THEY DONT EXIST")
    # load_data_set returns variables which get fed into model builder 
    # X, Y, uids = load_pyrenote_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, use_dump=True)
    
    # folder = 'train'
    # X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH,nonBird_labels, found)

    folder = 'Mixed_Bird-20220126T212121Z-003'
    X, Y, uids = load_pyrenote_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH,2)

    '''# print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
    # print(f'len of X {len(X)}')
    # bird1 = X[0] #data point, [0][0] feature value of dp, yes
    # print(bird1)
    # print(f'len of bird1 {len(bird1)}') # 216
    # print(f'shape of bird1 {bird1.shape}') # 216,72, r,c frequency bins, time bins
    # print(f'arrays inside bird1 {len(X[0][0])}') # 72
    # print(f'shape of bird1 first array {X[0][0].shape}')
    # print(f'bird1 uid {uids[0]}')
    # print(f'number of different birds {len(uids)}')
    # spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='mel', x_axis='time') # displays rotated
    # print(spec)
    # plt.show()
    # return'''


    print("\n")
    print("----------------------------------------------------------------------------------------------")
    # test_dataset = CustomAudioDataset(X, Y, uids)

    '''# pos, total = 0,0
    #remove green and red labels
    #for k in found:
        #print(k, found[k])'''

    # Prolonged time complexity
    # X, Y, uids =  random_split_to_fifty(X, Y, uids)

    '''# for y in Y:
    #     pos += sum(y)
    #     total += len(y)
    # print(pos, total, pos/total, len(Y))

    #features above feed into below

    #all_tags = create_tags(datasets_dir, folder)'''
    all_tags = [0,1]
    '''#print(len(Counter(all_tags)))
    #for c in range(10):
    #    print(Y[c])
    #return'''
    
    X_train, X_val, Y_train, Y_val, uids_train, uids_val = train_test_split(X, Y, uids, test_size=.3)

    X_val, X_test, Y_val, Y_test, uids_val, uids_test = train_test_split(X_val, Y_val, uids_val, test_size=.33)

    # print(X_train.shape, Y_train.shape, uids_train.shape)
    # print(X_val.shape, Y_val.shape, uids_val.shape)
    #create tensors
    # print(X_train.shape) #100, 216, 72
    # return 

    #if torch.cuda.is_available(): #get this to work, does not detect gpu. works on tweety env(slow)
    device = torch.cuda.device('cuda:0')
    # print(f'device {device.get_device_name()}')
    print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
    name = torch.cuda.get_device_name(device)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {name} ")# torch.cuda.get_device_name(0)
    
    #1) batch the X_train
    #For ex, batch_size=5 means that each batch is gonna be (5, 216, 72)
    #2) specify in the model that the input is (216, 72)

    X_train = torch.FloatTensor(X_train).to(torch.cuda.current_device())#.cuda()
    X_val = torch.FloatTensor(X_val).to(torch.cuda.current_device())#.cuda()

    Y_train = torch.LongTensor(Y_train).to(torch.cuda.current_device())#.cuda()
    Y_val = torch.LongTensor(Y_val).to(torch.cuda.current_device())#.cuda()

    X_test = torch.FloatTensor(X_test).to(torch.cuda.current_device())#.cuda()
    Y_test = torch.LongTensor(Y_test).to(torch.cuda.current_device())#.cuda()

    print(f'is X_train on GPU? {X_train.is_cuda}')
    print(f'is X_val on GPU? {X_val.is_cuda}')

    print(f'is Y_train on GPU? {Y_train.is_cuda}')
    print(f'is Y_val on GPU? {Y_val.is_cuda}')

    print(f'is X_test on GPU? {X_test.is_cuda}')
    print(f'is Y_test on GPU? {Y_test.is_cuda}')

    print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')

    #region
    # print('---------------------------------')
    # print('\n')
    # print(X_train)
    # print('\n')
    # print(X_val)
    # print('\n')
    # print(Y_train)
    # print('\n')
    # print(Y_val)
    # print('\n')
    # print('---------------------------------')

    #uids_train = torch.LongTensor(uids_train).cuda()
    #uids_val = torch.LongTensor(uids_val).cuda()
    #endregion
    
    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    #test_dataset = CustomAudioDataset(X_test[:6], Y_test[:6], uids_test[:6])

    X, Y, uid = train_dataset.__getitem__(0)

    #region
    # print(X.detach().cpu().numpy())
    # # return...
    # print('\n')
    # print(X[0].detach().cpu().numpy())
    # print('\n')
    # print(Y[0].detach().cpu().numpy())
    # print('\n')
    # print(uid)
    # print('\n')

    # bird1 = X[0].detach().cpu().numpy()
    # spec1 = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel')
    # plt.show()
    # return
    #endregion

    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)

    #region
    # X, Y, uid = val_dataset.__getitem__(0)
    # print('\n')
    # print(X[0].detach().cpu().numpy())
    # print('\n')
    # print(Y[0].detach().cpu().numpy())
    # print('\n')
    # print(uid)
    # print('\n')

    # bird2 = X[0].detach().cpu().numpy()
    # spec2 = librosa.display.specshow(bird2, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel')
    # plt.show()
    # return
    #endregion

    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)

    #region
    # X, Y, uid = test_dataset.__getitem__(0)
    # print('\n')
    # print(X[0].detach().cpu().numpy())
    # print('\n')
    # print(Y[0].detach().cpu().numpy())
    # print('\n')
    # print(uid)
    # print('\n')

    # bird3 = X[0].detach().cpu().numpy()
    # spec3 = librosa.display.specshow(bird3, hop_length = HOP_LENGTH,sr = SR, y_axis='time', x_axis='mel')
    # plt.show()
    # return

    # train_dataset.to(device)
    # val_dataset.to(device)

    # print(f'is train_dataset on GPU? {train_dataset.is_cuda}')
    # print(f'is val_dataset on GPU? {train_dataset.is_cuda}')

    # print('\n')
    # print('---------------------------------')
    # #print(train_dataset[:])
    # print('\n')
    # #print(val_dataset[:])
    # print('\n')
    # print('---------------------------------')

    # train_dataset = (train_dataset)
    # val_dataset = (val_dataset)
    #endregion

    return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR

def model_build(all_tags, n_mels, train_dataset, val_dataset, Skip, lr, batch_size, epochs, outdir):
    
    if Skip:
        for f in os.listdir(outdir):
            shutil.rmtree(os.path.join(outdir, f),ignore_errors=True)
        for f in os.listdir(outdir):
            os.remove(os.path.join(outdir, f))
    else:   
        pass
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd() 
    os.chdir(outdir)

    device = torch.cuda.device('cuda')
    print(f'device {torch.cuda.get_device_name(device)}')
    name = torch.cuda.get_device_name(device)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {name} ")# torch.cuda.get_device_name(0)
    
    model = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, 86), device, batchsize = batch_size, binary=False)
    # model = 
    # model.cuda()
    model.to(torch.cuda.current_device())#()
    model.train()
    # summary(model.to(torch.cuda.current_device()),(1, n_mels, 216))

    # print(f'is model in GPU? {model.is_cuda}')

    print(torch.cuda.get_device_name(model.device))
    for i in model.parameters():
        # print('model param')
        # print(i)
        print(i.is_cuda)
    next(model.parameters()).to(torch.cuda.current_device())#.cuda()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    print(dt_string)

    train_dataset = train_dataset #(train_dataset).cuda()
    val_dataset = val_dataset#(val_dataset).cuda()

    history, test_out, start_time, end_time, date_str = model.train_pipeline(train_dataset,val_dataset, None,
                                                                       lr=lr, batch_size=batch_size,epochs=epochs, save_me=True,
                                                                       fine_tuning=False, finetune_path=None, outdir=outdir)#.cuda()
    print("Training time:", end_time-start_time)

    os.chdir(cwd)

    with open(os.path.join(outdir,"nips_history.pkl"), 'wb') as f:   # where does this go??? it has to end up in data/out
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL) 

    return model, date_str

def evaluate(model,test_dataset, date_str, hop_length, sr, outdir,temporal_graphs): # How can we evaluauate on a specific wav file though?? and show time in the csv? and time on a spectrorgam? ¯\_(ツ)_/¯
    
    model_weights = os.path.join(outdir,f"model_weights-{date_str}.h5") # time sensitive file title
    tweetynet = model
    test_out, time_segs = tweetynet.test_load_step(test_dataset, hop_length, sr, model_weights=model_weights) 

    #process the predictions here?
    test_out.to_csv(os.path.join(outdir,"Evaluation_on_data.csv"))

    time_segs.to_csv(os.path.join(outdir,"Time_intervals.csv"))
    
    # orig_stdout = sys.stdout

    # sys.stdout = open(os.path.join('data/out','file_score_rates.txt'), 'w')
    file_score(temporal_graphs)
    #process the predictions here?
    # sys.stdout.close()

    # sys.stdout = orig_stdout
    # file_graph_temporal(temporal_graphs) 
    # file_graph_temporal_rates(temporal_graphs) 
    
    return print("Finished Classifcation")