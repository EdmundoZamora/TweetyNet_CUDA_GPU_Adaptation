import os
import sys
import csv
import math
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
# from torchsummary import summary

#from microfaune.audio import wav2spc, create_spec, load_wav
from TweetyNetAudio import wav2spc, create_spec, load_wav
import random
import librosa


def get_frames(x, frame_size, hop_length):
    return ((x) / hop_length) + 1 #(x - frame_size)/hop_length + 1
def frames2seconds(x, sr):
    return x/sr
def find_tags(data_path, folder):
    fnames = os.listdir(os.path.join(data_path, "temporal_annotations_nips4b"))
    csvs = []
    for f in fnames:
        #print(f)
        csvs.append(pd.read_csv(os.path.join(data_path, "temporal_annotations_nips4b", f), index_col=False, names=["start", "duration", "tag"]))
    return csvs

# works
def find_pyrenote_tags(data_path, folder):
    # fnames = os.listdir(os.path.join(data_path, "temporal_annotations_nips4b"))
    Pyre = pd.read_csv(os.path.join(data_path, "for_data_science_newline_fixed.csv"), index_col=False, usecols=["IN FILE","OFFSET", "DURATION", "MANUAL ID","SAMPLE RATE"])

    # for_data_science_newline_fixed.csv
    # csvs = []
    # for f in fnames:
        #print(f)
        # csvs.append(pd.read_csv(os.path.join(data_path, "temporal_annotations_nips4b", f), index_col=False, names=["OFFSET", "DURATION", "MANUAL ID"]))
        # instead of doing this we just get that info from one dataframe.! WIP
    Pyre = Pyre[Pyre["SAMPLE RATE"] == 44100]
    # print(Pyre)
    # return
    return Pyre
# works
def create_pyrenote_tags(data_path, folder):

    csv = find_pyrenote_tags(data_path, folder) # one dataframe
    # print()

    tags = csv["MANUAL ID"] # tags column

    tag = [] 
    for t in tags:# for each column/series
        # for a in t: # for each species in the individual column/series
        tag.append(t)
    tag = set(tag) # remove duplicates
    # tags = {"None": 0} # dictionary counting the number of species
    tags = {}
    for i, t in enumerate(sorted(tag)): # adding to dictionary and updating species counts
        tags[t] = i + 1
    # print(tags)
    return tags # returns a dictionary of species and their counts
# works
def compute_pyrenote_feature(data_path, folder, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    
    features = {"uids": [], "X": [], "Y": [], "time_bins": []}
    '''
    # print(data_path)
    # folder = 
    # print(os.path.join(data_path, folder))
    # cwd = os.getcwd() 
    # print(cwd)
    '''
    pyre_notes = find_pyrenote_tags(data_path,folder)
    valid_filenames = pyre_notes["IN FILE"].drop_duplicates().values.tolist() 
    '''
    # print(valid_filnames)
    # return
    '''
    file_path = os.path.join(data_path,"Mixed_Bird-20220126T212121Z-003","Mixed_Bird")
    # print(file_path)
    filenames = os.listdir(file_path)
    # print(filenames)
    true_wavs = [i for i in filenames if i in valid_filenames] # keep in mind not all wavs have been downloaded yet.
    
    #region
    # print(os.path.join(data_path,'birdwavs.txt'))
    #C:\Users\lianl\Repositories\Q1-Project-Code\data\PYRE\ birdwavs.txt
    # with open(os.path.join(data_path,'birdwavs.txt'), 'w') as filehandle: # for getting the stats in this set of data., makes an extra new line in txt.
    #     for listitem in true_wavs:
    #         filehandle.write('%s\ n' % listitem)
    # return 
    # print(true_wavs)
    # print(len(true_wavs))
    # print(type(true_wavs))
     # filter out the non-44100 sampling rates frequencies
    # return
    "Recordings in the format of nips4b_birds_{folder}filexxx.wav"
    
    "annotations in the format annotation_{folder}xxx.csv"
    #endregion
    
    tags = create_pyrenote_tags(data_path, folder)
    '''
    # print(tags)
    # return
    ''' 
    
    for f in true_wavs:
        # Y = compute_pyrenote_Y(wav,f, spc, tags, data_path, folder, SR, frame_size, hop_length) # fix this
        # print(computed*(Y.shape[0]//computed))
        print(f)
        wav = os.path.join(file_path, f)
        spc,len_audio = wav2spc(wav, fs=SR, n_mels=n_mels) # returns array for display melspec (216,72)
        time_bins = len_audio/spc.shape[1] # number of seconds in 1 time_bin
        Y = compute_pyrenote_Y(wav,f, spc, tags, data_path, folder, SR, frame_size, hop_length) # fix this

        features["uids"].append(f)#.extend([f]*freq_axis) # need 31 of f
        features["X"].append(spc)#.extend(spc_split)#.append(spc)
        features["Y"].append(Y)#.extend(Y_split)#.append(Y)
        features["time_bins"].append(time_bins)
    return features
# works
def compute_pyrenote_Y(wav, f, spc, tags, data_path, folder, SR, frame_size, hop_length):
    # file_num = f.split("file")[-1][:3]
    # print(wav)
    # print(os.path.isfile(wav))
    infile = f # wav file name
    # print(f)
    Pyre_notes = find_pyrenote_tags(data_path, folder)

    wav_notes = Pyre_notes[Pyre_notes['IN FILE'] == f ]
    # print(wav_notes)
    # print(type(wav_notes))
    # print(wav_notes.shape)
    # return

    if os.path.isfile(wav):

        x, sr = librosa.load(wav, sr=SR)
        # print(x)
        # print(sr)
        # return

        annotation = wav_notes[['OFFSET','DURATION','MANUAL ID']].reset_index(drop = True)#pd.read_csv(Pyre_notes, index_col=False, names=["start", "duration", "tag"])
        # print(annotation)
        # print(type(annotation))
        # print(annotation.shape)
        # return

        y = calc_pyrenote_Y(x, sr, spc, annotation, tags, frame_size, hop_length)
        # return

        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]
# works
def calc_pyrenote_Y(x, sr, spc, annotation, tags, frame_size, hop_length):
    y = [0] * spc.shape[1] # array of zeros
    # print(y)
    # return
    for i in range(len(annotation)):
        # print(len(annotation))
        # return
        # print(i)
        # print(annotation.loc[i, "OFFSET"])
        # return
        # print(annotation.loc[i, "OFFSET"] * sr)
        # return
        start = get_frames(annotation.loc[i, "OFFSET"] * sr, frame_size, hop_length)
        # print(start)
        # return
        end = get_frames((annotation.loc[i, "OFFSET"] + annotation.loc[i, "DURATION"]) * sr, frame_size, hop_length)
        # print(end)
        # return
        #print(annotation["tag"], len(annotation["tag"]))

        # if annotation["tag"][i] not in nonBird_labels:
        for j in range(math.floor(start), math.floor(end)): #CORRECT WAY TO ADD TRUE LABELS?
            # print(f'spc shape {spc.shape}') # (72, 1)
            # print(f'Manual ID{annotation.loc[0, "MANUAL ID"]}') # Alopochelidon fucata Tawny-headed Swallow
            # print(f'length of labels {len(y)}') #1
            # print(f'indexing label {j}')      #81 which file? maybe the wav_file has not been annotated?
            y[j] = 1 # For binary use. add if statement later tags[annotation.loc[i, "tag"]]
        # else: 
        #     #print(str(annotation["tag"][i]))
        #     found[str(annotation["tag"][i])] += 1
        # print(y) #COMPARE TO SPECTROGRAM Accipiter-superciliosus-329800.wav
        # return
    return y
# works
def load_pyrenote_dataset(data_path, folder, SR, n_mels, frame_size, hop_length, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_dataset.pkl".format(folder))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_pyrenote_feature(data_path, folder, SR, n_mels, frame_size, hop_length)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    # print(f'dataset is {dataset}')
    # print(dataset)
    #this section here has me confused, rotates the spectrograms, microfaune implementation.
    # inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 216]
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] >= 0] # consider monophonic, start windowing
    # inds = [i for i, x in enumerate(dataset["X"])]
    # X = np.array([dataset["X"][i].transpose() for i in inds]).astype(np.float32)/255

    # X = np.array([np.rot90(dataset["X"][i].astype(np.float32)/255,3) for i in inds], dtype=object)#.astype(np.float32)/255
    # X = np.array([np.rot90(dataset["X"][i],3) for i in inds]).astype(np.float32)/255 rotation causes frequency prediction outputs instead of timebins
    # X = np.array([dataset["X"][i] for i in inds]).astype(np.float32)/255
    # X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    # Y = np.array([dataset["Y"][i] for i in inds]).astype(np.longlong)
    # uids = np.array([dataset["uids"][i] for i in inds])
    X = dataset['X']
    Y = dataset['Y']
    uids = dataset['uids']
    time_bins = dataset['time_bins']

    return X, Y, uids, time_bins

def load_pyrenote_splits(spcs, ys, uids, time_bins, windowsize, data_path, folder, set_type, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_{}.pkl".format(folder, set_type))
    print(f"loading dataset for {set_type}")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else: # need to go through each element
        dataset = window_data(spcs, ys, uids, time_bins, windowsize)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = np.array([dataset["X"]]).astype(np.float32)/255
    X = X.reshape(X.shape[1], 1, X.shape[2], X.shape[3])
    Y = np.array([dataset["Y"]]).astype(np.longlong)
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    uid = np.array([dataset["uids"]])
    uid = uid.reshape(uid.shape[1])
    return X, Y, uid

def window_data(spcs, ys, uids, time_bins, windowsize):
    windowed_dataset = {"uids": [], "X": [], "Y": []}
    print("Windowing Spectrogram")
    for i in range(len(uids)):
        spc_split, Y_split, uid_split = window_spectrograms(spcs[i],ys[i], uids[i], time_bins[i], windowsize)
        windowed_dataset["X"].extend(spc_split)
        windowed_dataset["Y"].extend(Y_split)
        windowed_dataset["uids"].extend(uid_split)
    return windowed_dataset

def window_spectrograms(spc, Y, uid, time_bin, windowsize):
    computed = windowsize//time_bin #verify, big assumption. are time bins consistant?
    # print(computed*(Y.shape[0]//computed))
    time_axis = int(computed*(Y.shape[0]//computed))
    freq_axis = int(Y.shape[0]//computed) # 31, 2, 19
    spc_split = np.split(spc[:,:time_axis],freq_axis,axis = 1)
    Y_split = np.split(Y[:time_axis],freq_axis)
    uid_split = [str(i) + "_" + uid for i in range(freq_axis)]
    return spc_split, Y_split, uid_split

def create_tags(data_path, folder):
    csvs = find_tags(data_path, folder)
    tags = [csv["tag"] for csv in csvs]
    tag = []
    for t in tags:
        for a in t:
            tag.append(a)
    tag = set(tag)
    tags = {"None": 0}
    for i, t in enumerate(sorted(tag)):
        tags[t] = i + 1
    return tags

def compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    
    features = {"uids": [], "X": [], "Y": []}
    
    filenames = os.listdir(os.path.join(data_path, folder))
    
    "Recordings in the format of nips4b_birds_{folder}filexxx.wav"
    
    "annotations in the format annotation_{folder}xxx.csv"
    
    tags = create_tags(data_path, folder)
    
    for f in filenames:
		#signal, SR = downsampled_mono_audio(signal, sample_rate, SR)
        spc, _ = wav2spc(os.path.join(data_path, folder, f), fs=SR, n_mels=n_mels)
        Y = compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length, nonBird_labels, found)
        features["uids"].append("0_"+f)
        features["X"].append(spc)
        features["Y"].append(Y)
    return features

def compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length, nonBird_labels, found):
    file_num = f.split("file")[-1][:3]
    fpath = os.path.join(data_path, "temporal_annotations_nips4b", "".join(["annotation_", folder, file_num, ".csv"]))
    if os.path.isfile(fpath):
        x, sr = librosa.load(os.path.join(data_path, folder, f), sr=SR)
        annotation = pd.read_csv(fpath, index_col=False, names=["start", "duration", "tag"])
        y = calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length, nonBird_labels, found)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]

def calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length, nonBird_labels, found):
    y = [0] * spc.shape[1]
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "start"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "start"] + annotation.loc[i, "duration"]) * sr, frame_size, hop_length)
        #print(annotation["tag"], len(annotation["tag"]))
        if annotation["tag"][i] not in nonBird_labels:
            for j in range(math.floor(start), math.floor(end)):
                y[j] = 1 # For binary use. add if statement later tags[annotation.loc[i, "tag"]]
        else: 
            #print(str(annotation["tag"][i]))
            found[str(annotation["tag"][i])] += 1
    return y

def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :] # questionable
    Y_train, Y_test = Y[ind_train], Y[ind_test] # questionable
    return ind_train, ind_test

def get_pos_total(Y):
    pos, total = 0,0
    for y in Y:
        pos += sum(y)
        total += len(y)
    #print(pos, total, pos/total, len(Y))
    return pos, total

def random_split_to_fifty(X, Y, uids):
    pos, total = get_pos_total(Y)
    j = 0
    while (pos/total < .50):
        idx = random.randint(0, len(Y)-1)
        if (sum(Y[idx])/Y.shape[1] < .5):
            #print(uids[idx],(sum(Y[idx])/Y.shape[1]))
            X = np.delete(X, idx, axis=0)
            Y = np.delete(Y, idx, axis=0)
            uids = np.delete(uids, idx, axis=0)
            #print(j, pos/total)
            j += 1

        pos, total = get_pos_total(Y)
    return X, Y, uids

def load_dataset(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_dataset.pkl".format(folder))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)

    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 216]
    # X = np.array([dataset["X"][i].transpose() for i in inds]).astype(np.float32)/255 # tune
    # X = np.array([dataset["X"][i] for i in inds]).astype(np.float32)/255
    X = np.array([dataset["X"][i] for i in inds]).astype(np.float32)/255 # over training, with norm label, .005 lr, 5, bs, 500 E
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    Y = np.array([dataset["Y"][i] for i in inds]).astype(np.longlong)
    uids = np.array([dataset["uids"][i] for i in inds])

    # to tensor
    # X = torch.LongTensor(X).cuda()
    # Y = torch.LongTensor(Y).cuda()
    # uids = torch.LongTensor(uids).cuda()

    return X, Y, uids

def load_splits(spcs, ys, uids, data_path, folder, set_type, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_{}.pkl".format(folder, set_type))
    print(f"loading dataset for {set_type}")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else: # need to go through each element
        dataset = {"X": spcs, "Y": ys, "uids": uids}
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    X = np.array([dataset["X"]])#.astype(np.float32)/255
    X = X.reshape(X.shape[1], 1, X.shape[3], X.shape[4])
    Y = np.array([dataset["Y"]])#.astype(np.longlong)
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    uid = np.array([dataset["uids"]])
    uid = uid.reshape(uid.shape[1])
    print(X.shape, Y.shape, uid.shape)
    return X, Y, uid