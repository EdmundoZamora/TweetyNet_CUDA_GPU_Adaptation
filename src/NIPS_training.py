import os
import sys
import csv
import math
import pickle
import shutil
from collections import Counter
from datetime import datetime

from scoring import*
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

from Load_data_functions import load_dataset, load_pyrenote_dataset, load_pyrenote_splits, load_splits

def apply_features(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, window_size, dataset):
    train = True
    fineTuning = False
    print("----------------------------------------------------------------------------------------------")
    print("\n")
    print("WAV FILES - THEY EXIST")
    # load_data_set returns variables which get fed into model builder 
    if dataset == "NIPS":
        folder = 'train'
        X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, use_dump=True)
        #print(f'X shape {X.shape}') #number of birds, rows of each data column of each data.
        #print(f'len of X {len(X)}')
        #bird1 = X[0] #data point, [0][0] feature value of dp, yes
        #bird1 = bird1.reshape(bird1.shape[1], bird1.shape[2])
        #print(bird1)
        #print(f'len of bird1 {len(bird1)}') # 216
        #print(f'shape of bird1 {bird1.shape}') # 216,72, r,c frequency bins, time bins
        #print(f'arrays inside bird1 {len(X[0][0])}') # 72
        #print(f'shape of bird1 first array {X[0][0].shape}')
        #print(f'bird1 uid {uids[0]}')
        #print(f'number of different birds {len(uids)}')
        #spec = librosa.display.specshow(bird1, hop_length = HOP_LENGTH,sr = SR, y_axis='mel', x_axis='time') # displays rotated
        #print(spec)
        #plt.show()
        X_train, X_val, Y_train, Y_val, uids_train, uids_val = train_test_split(X, Y, uids, test_size=.3) # seed = 0
        X_val, X_test, Y_val, Y_test, uids_val, uids_test = train_test_split(X_val, Y_val, uids_val, test_size=.66)
        
        X_train, Y_train, uids_train = load_splits(X_train, Y_train, uids_train, datasets_dir, folder, "train")
        X_val, Y_val, uids_val = load_splits(X_val, Y_val, uids_val, datasets_dir, folder, "val")
        X_test, Y_test, uids_test = load_splits(X_test, Y_test, uids_test, datasets_dir, folder, "test")
        all_tags = [0,1]

        device = torch.cuda.device('cuda:0')
        # print(f'device {device.get_device_name()}')
        print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
        name = torch.cuda.get_device_name(device)
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

        train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
        val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
        test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
        return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR
        
    elif dataset == "PYRE":
        folder = "Mixed_Bird-20220126T212121Z-003"
        X, Y, uids, time_bins = load_pyrenote_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH)
        all_tags = [0,1]
        # need
        #Split by file make CPU and GPU have the same files in splits though
        pre_X_train, pre_X_val, pre_Y_train, pre_Y_val, pre_uids_train, pre_uids_val, pre_time_bins_train, pre_time_bins_val = train_test_split(X, Y, uids, time_bins, test_size=.3) # Train 70% Val 30%

        pre_X_val, pre_X_test, pre_Y_val, pre_Y_test, pre_uids_val, pre_uids_test, pre_time_bins_val, pre_time_bins_test= train_test_split(pre_X_val, pre_Y_val, pre_uids_val, pre_time_bins_val, test_size=.66)# val 10%, test 20%

        #window spectrograms
        X_train, Y_train, uids_train, = load_pyrenote_splits(pre_X_train, pre_Y_train, pre_uids_train, pre_time_bins_train, window_size, datasets_dir, folder, "train")
        X_val, Y_val, uids_val, = load_pyrenote_splits(pre_X_val, pre_Y_val, pre_uids_val, pre_time_bins_val, window_size, datasets_dir, folder, "val")
        X_test, Y_test, uids_test, = load_pyrenote_splits(pre_X_test, pre_Y_test, pre_uids_test, pre_time_bins_test, window_size, datasets_dir, folder, "test")

        device = torch.cuda.device('cuda:0')
        # print(f'device {device.get_device_name()}')
        print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
        name = torch.cuda.get_device_name(device)
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

        train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
        val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
        test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
        return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR
    else:
        print(f"dataset:{dataset} does not exist")
        return None

    # device = torch.cuda.device('cuda:0')
    # # print(f'device {device.get_device_name()}')
    # print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
    # name = torch.cuda.get_device_name(device)
    # print(f"Using {name} ")# torch.cuda.get_device_name(0)
    
    # #1) batch the X_train
    # #For ex, batch_size=5 means that each batch is gonna be (5, 216, 72)
    # #2) specify in the model that the input is (216, 72)

    # X_train = torch.FloatTensor(X_train).to(torch.cuda.current_device())#.cuda()
    # X_val = torch.FloatTensor(X_val).to(torch.cuda.current_device())#.cuda()

    # Y_train = torch.LongTensor(Y_train).to(torch.cuda.current_device())#.cuda()
    # Y_val = torch.LongTensor(Y_val).to(torch.cuda.current_device())#.cuda()

    # X_test = torch.FloatTensor(X_test).to(torch.cuda.current_device())#.cuda()
    # Y_test = torch.LongTensor(Y_test).to(torch.cuda.current_device())#.cuda()

    # print(f'is X_train on GPU? {X_train.is_cuda}')
    # print(f'is X_val on GPU? {X_val.is_cuda}')

    # print(f'is Y_train on GPU? {Y_train.is_cuda}')
    # print(f'is Y_val on GPU? {Y_val.is_cuda}')

    # print(f'is X_test on GPU? {X_test.is_cuda}')
    # print(f'is Y_test on GPU? {Y_test.is_cuda}')

    # print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
    
    # train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    # val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)
    # test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    # return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR

    # X, Y, uid = train_dataset.__getitem__(0)

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
    # return all_tags, n_mels, train_dataset, val_dataset, test_dataset, HOP_LENGTH, SR

def model_build( all_tags, n_mels, train_dataset, val_dataset, Skip, time_bins, lr, batch_size, epochs, outdir, ):
    
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
    
    model = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, time_bins), time_bins, device, binary = False)
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

    history, start_time, end_time, date_str = model.train_pipeline(train_dataset,val_dataset, 
                                                                       lr=lr, batch_size=batch_size,epochs=epochs, save_me=True,
                                                                       fine_tuning=False, finetune_path=None, outdir=outdir)#.cuda()
    print("Training time:", end_time-start_time)

    os.chdir(cwd)

    with open(os.path.join(outdir,"nips_history.pkl"), 'wb') as f:   # where does this go??? it has to end up in data/out
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL) 

    return model, date_str

def evaluate(model,test_dataset, date_str, hop_length, sr, outdir,temporal_graphs,window_size): # How can we evaluauate on a specific wav file though?? and show time in the csv? and time on a spectrorgam? ¯\_(ツ)_/¯
    
    model_weights = os.path.join(outdir,f"model_weights-{date_str}.h5") # time sensitive file title
    tweetynet = model
    test_out, time_segs = tweetynet.test_load_step(test_dataset, hop_length, sr, model_weights=model_weights,window_size = window_size) 

    #process the predictions here?
    test_out.to_csv(os.path.join(outdir,"Evaluation_on_data.csv"))

    time_segs.to_csv(os.path.join(outdir,"Time_intervals.csv"))
    
    # orig_stdout = sys.stdout

    # sys.stdout = open(os.path.join('data/out','file_score_rates.txt'), 'w')
    file_score(temporal_graphs)
    #process the predictions here?
    # sys.stdout.close()

    # sys.stdout = orig_stdout
    file_graph_temporal(temporal_graphs) 
    file_graph_temporal_rates(temporal_graphs) 
    
    return print("Finished Classifcation")