import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import librosa
from torch import cuda
from tqdm import tqdm
#from network import TweetyNet
from torch.utils.data import DataLoader
from network import TweetyNet
from torch import nn
from torch.nn import functional as F
from EvaluationFunctions import frame_error, syllable_edit_distance
#from microfaune.audio import wav2spc, create_spec, load_wav
from TweetyNetAudio import wav2spc, create_spec, load_wav, get_time
from CustomAudioDataset import CustomAudioDataset
from datetime import datetime


"""
Helper Functions to TweetyNet so it feels more like a Tensorflow Model.
This includes instantiating the model, training the model and testing. 
"""
class TweetyNetModel(nn.Module):
    # Creates a tweetynet instance with training and evaluation functions.
    # input: num_classes = number of classes TweetyNet needs to classify
    #       input_shape = the shape of the spectrograms when fed to the model.
    #       ex: (1, 1025, 88) where (# channels, # of frequency bins/mel bands, # of frames)
    #       device: "cuda" or "cpu" to specify if machine will run on gpu or cpu.
    # output: None
    def __init__(self, num_classes, input_shape, window_size, device = torch.cuda.device('cuda'), epochs = 1,batchsize = 32, binary=False, criterion=None, optimizer=None):
        super(TweetyNetModel, self).__init__()

        self.model = TweetyNet(num_classes=num_classes,
                               input_shape=input_shape,
                               padding='same',
                               conv1_filters=32,
                               conv1_kernel_size=(5, 5),
                               conv2_filters=64,
                               conv2_kernel_size=(5, 5),
                               pool1_size=(8, 1),
                               pool1_stride=(8, 1),
                               pool2_size=(8, 1),
                               pool2_stride=(8, 1),
                               hidden_size=None,
                               rnn_dropout=0., #0.3
                               num_layers=1
                               )

        self.device = torch.cuda.current_device()
        self.model.to(self.device)
        self.binary = binary
        self.window_size = window_size
        self.runtime = 0
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(params=self.model.parameters())
        self.epochs = epochs
        self.batchsize = batchsize
        self.n_train_examples = self.batchsize *30 
        self.n_valid_examples = self.batchsize *10 
        

    """
    Function: print_results
    Input: history is a dictionary of the loss, accuracy, and edit distance at each epoch
    output: None
    purpose: Print the results from training
    """
    @staticmethod
    def print_results(history, show_plots=False, save_plots=True):
        plt.figure(figsize=(9, 6))
        plt.title("Loss")
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.legend(["loss", "val_loss"])
        if save_plots:
            plt.savefig('loss.png')
        if show_plots:
            plt.show()

        plt.figure(figsize=(9, 6))
        plt.title("Accuracy")
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.legend(["acc", "val_acc"])
        if save_plots:
            plt.savefig('acc.png')
        if show_plots:
            plt.show()

        # plt.figure(figsize=(9, 6))
        # plt.title("Edit Distance")
        # plt.plot(history["edit_distance"])
        # plt.plot(history["val_edit_distance"])
        # plt.legend(["edit_distance", "val_edit_distance"])
        # if save_plots:
        #     plt.savefig('edit_distance.png')
        # if show_plots:
        #     plt.show()

    def reset_weights(self):
        for name, module in self.model.named_children():
            if hasattr(module, 'reset_parameters'):
                print('resetting ', name)
                module.reset_parameters()

    """
    Function: train_pipeline
    Input: the datasets used for training, validation, testing, and hyperparameters
    output: history is the loss, accuracy, and edit distance at each epoch, any test predictions
        the model made, and the duration of the training.
    purpose: Set up training TweetyNet, to save weights, and test the model after training
    """
    def train_pipeline(self, train_dataset, val_dataset=None, lr=.005, batch_size=64,
                       epochs=100, save_me=True, fine_tuning=False, finetune_path=None, outdir=None):
        
        if fine_tuning:
            self.model.load_weights(finetune_path)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = None

        if val_dataset != None:
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=int(len(train_data_loader)),
                                                        epochs=epochs,
                                                        anneal_strategy='linear')
        start_time = datetime.now()

        # cwd = os.getcwd() # might have to be inherited from runfile
        # os.chdir(outdir)

        history = self.training_step(train_data_loader, val_data_loader, scheduler, epochs)

        #history = history.cpu()

        end_time = datetime.now()
        self.runtime = end_time - start_time

        if save_me: # save to temp?
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            # torch.save(self.model.state_dict(), os.path.join(outdir,f"model_weights-{date_str}.h5"))
            torch.save(self.model.state_dict(), f"model_weights-{date_str}.h5")
            
        # cwd = os.getcwd() # might have to be inherited from runfile
        # os.chdir(outdir) 
        self.print_results(history)  # save to out, saves to wd. works on cpu. get it to work on GPU
        #os.chdir(cwd)
        return history, start_time, end_time, date_str



    """
    Function: training_step
    Input: train_loader is the training dataset, val_loader is the validation dataset
        The scheduler is used to make the learning rate dynamic and epochs are the number of 
        epochs to train for.
    output: history which is the loss, accuracy and edit distance
    purpose: to train TweetyNet.
    """
    def training_step(self, train_loader, val_loader, scheduler, epochs):
        history = {"loss": [],
                   "val_loss": [],
                   "acc": [],
                   "val_acc": [],
                #    "edit_distance": [],
                #    "val_edit_distance": [],
                   "best_weights" : 0
                   }
        #add in early stopping criteria and saving best weights at each epoch
        #Added saving best model weights to validation step

        for e in range(epochs):  # loop over the dataset multiple times
            print("Start of epoch:", e)
            self.model.train(True)
            running_loss = 0.0
            correct = 0.0
            edit_distance = 0.0
            
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for i, data in loop: # train loader is custom audiodataset, getitem, for spectrogram.
                inputs, labels, uid = data
                #print(type(input))
                #print(type(labels))
                # print(f'before reshape{inputs.shape}') #tutor
                # print(inputs.shape)
                # if e == 0:
                #     bird = inputs[0].detach().cpu().numpy()
                #     bird = bird.reshape(inputs.shape[2], inputs.shape[3])
                #     print(uid[0])
                #     print(labels[0].detach().cpu().numpy())
                #     spec = librosa.display.specshow(bird,sr = 44100, hop_length = 1024, y_axis='time', x_axis='mel')
                #     plt.show()
                # else:
                #     pass                #return 
                # print(bird)
                # print(inputs[0])
                # return ...
        
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()# curr

                # labels = labels.to(self.device)
                #print(inputs.shape[0])
                #print(labels.shape[0])

                output = self.model(inputs)   # ones and zeros, temporal bird annotations.
                # print(f'output shape in train {output.shape}')
                # output = output.reshape(labels.shape[0], labels.shape[1]) #tutor
                # print(output.shape)
                # return
                # 5,216,72
                # print(output)
                #if self.binary:
                #    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))

                # print(output.requires_grad) #tutor

                loss = self.criterion(output, labels)

                loss.backward()

                # print(output.requires_grad)
                # self.optimizer.zero_grad() # tutor
                self.optimizer.step()
                scheduler.step()

                running_loss += loss.item()

                output = torch.max(output, dim=1)[1]#.squeeze()
        
                correct += (output == labels).float().sum().cpu().detach().numpy()

                # for j in range(len(labels)):
                    # edit_distance += syllable_edit_distance(output[j], labels[j])'''

                # print update Improve this to make it better Maybe a global counter
                #if i % 10 == 9:  # print every 10 mini-batches
                #    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss ))

                loop.set_description(f"Epoch [{e}/{epochs}]")
                loop.set_postfix(loss = loss.item(), acc = 100*float((output == labels).float().sum())/(len(data[1]) * self.model.input_shape[-1]))
                    
            history["loss"].append(running_loss)
            history["acc"].append(100 * correct / (len(train_loader.dataset) * self.window_size))
            
            print("Running Accuracy: ", 100 * float(correct / (len(train_loader.dataset) * self.model.input_shape[-1])))
            print("Running Loss: ", running_loss)
            # history["edit_distance"].append(edit_distance / (len(train_loader.dataset) * self.window_size))
            
            if val_loader != None:
                self.validation_step(val_loader, history) # , scheduler
        print('Finished Training')
        return history

    """
    Function: validation_step
    Input: val_loader is the validation dataset and the history is a dictionary to keep track of 
            Loss, accuracy, and edit distance
    output: None
    purpose: To validate TweetyNet at each epoch. Also saves the best model_weights at each epoch.
    """
    def validation_step(self, val_loader, history):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0.0
            val_edit_distance = 0.0
            for i, data in enumerate(val_loader):
                inputs, labels, uid = data
                # if i == 0:
                #     bird = inputs[0].detach().cpu().numpy()
                #     bird = bird.reshape(inputs.shape[2], inputs.shape[3])
                #     print(uid[0])
                #     print(labels[0].detach().cpu().numpy())
                #     spec = librosa.display.specshow(bird,sr = 44100, hop_length = 1024, y_axis='time', x_axis='mel')
                #     plt.show()
                # else:
                #     pass

            
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # self.optimizer.zero_grad()

                output = self.model(inputs)

                loss = self.criterion(output, labels)

                # loss.backward()
                # self.optimizer.step()
                # scheduler.step()
                
                val_loss += loss.item() #curr

                output = torch.max(output, dim=1)[1]  # gets indices of correct predictions of timebins in windows
                
                val_correct += (output == labels).float().sum().cpu().detach().numpy()

                '''# for j in range(len(labels)):
                #     val_edit_distance += syllable_edit_distance(output[j], labels[j])'''

            history["val_loss"].append(val_loss)
            history["val_acc"].append(100 * val_correct / (len(val_loader.dataset) * self.window_size))
            # history["val_edit_distance"].append(val_edit_distance / (len(val_loader.dataset) * self.window_size))


            print("Running Validation Accuracy: ", 100 * float(val_correct / (len(val_loader.dataset) * self.model.input_shape[-1])))
            print("Running Validation Loss: ", val_loss)

            if history["val_acc"][-1] > history["best_weights"]:
                torch.save(self.model.state_dict(), "best_model_weights.h5")
                history["best_weights"] = history["val_acc"][-1]



    """
    Function: testing_step
    Input: test_loader is the test dataset
    output: predictins that the model made
    purpose: Evaluate our model on a test set
    """
    def testing_step(self, test_loader, hop_length, sr, window_size):
        predictions = pd.DataFrame()
        self.model.eval()

        st_time = []
        dataiter = iter(test_loader)
        label, _, _ = dataiter.next()
        label = label.cpu().detach().numpy()

        #print(label.shape)
        for i in range(label.shape[-1]): # will change to be more general, does it only for one trainfile?
            st_time.append(get_time(i, hop_length, sr))
        st_time = np.array(st_time)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels, uids = data
                #print(type(labels))
                # print(labels)
                # if i == 0:
                #     bird = inputs[0].detach().cpu().numpy()
                #     bird = bird.reshape(inputs.shape[2], inputs.shape[3])
                #     print(uids[0])
                #     print(labels[0].detach().cpu().numpy())
                #     spec = librosa.display.specshow(bird,sr = 44100, hop_length = 1024, y_axis='time', x_axis='mel')
                #     plt.show()
                # else:
                #     pass

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # print(f'labels shape {labels.shape}')
                # print('Testing Step')

                output = self.model(inputs) # what is this output look like?, specify batch size here?


                '''# output = output.reshape(labels.shape[0], labels.shape[1]) # tutor
                # print(f'Testing step output shape {output.shape}') # 100, value (torch.Size([41,2,216]))
                # return 
                #print(output)
                #print(type(labels))'''
                temp_uids = []
                files = []
                window_file = []
                window_number = []
                frame_number = []
                overall_frame_number = []
                st_batch_times = []
                if self.binary: # weakly labeled
                    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))
                    temp_uids = np.array([[x] * output.shape[-1] for x in uids])
                    files.append(u)
                else:  # in the case of strongly labeled data
                    for u in uids:
                        st_batch_times.extend(st_time + (window_size*int(u.split("_")[0])))
                        for j in range(output.shape[-1]):
                             temp_uids.append(str(j + (output.shape[-1]*int(u.split("_")[0]))) + "_" + u)
                             window_file.append(u)
                             frame_number.append(j)
                             overall_frame_number.append(j+ (output.shape[-1]*int(u.split("_")[0])))
                             window_number.append(int(u.split("_")[0]))
                             files.append("_".join(u.split("_")[1:]))
                    temp_uids = np.array(temp_uids)
                    window_file = np.array(window_file)
                    window_number = np.array(window_number)
                    frame_number = np.array(frame_number)
                    overall_frame_number = np.array(overall_frame_number)
                    st_batch_times = np.array(st_batch_times)
                labels = labels.cpu().detach().numpy()

                zero_pred = output[:, 0, :].cpu().detach().numpy()# curr
                one_pred = output[:, 1, :].cpu().detach().numpy()# curr

                # Are we getting the correct prediction? in the sense that its the predictions calculated not something else?
                # pred = torch.max(output, dim=1)[1].squeeze().cpu().detach().numpy() # causing problems
                pred = torch.max(output, dim=1)[1].cpu().detach().numpy()
                d = {"uid": temp_uids.flatten(), "window file": window_file.flatten(), "file":files, 
                        "overall frame number": overall_frame_number, "frame number": frame_number, "window number": window_number, 
                        "zero_pred": zero_pred.flatten(), "one_pred": one_pred.flatten(), 
                        "pred": pred.flatten(),"label": labels.flatten(), "temporal_frame_start_times": st_batch_times.flatten()}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)

                #process the predictions here?


                '''# pred = output.cpu().detach().numpy() # tutor
                # print(pred) # tutor

                # print(type(pred))
                # print(pred.dtype)
                # return
                #pred = longtensor.numpy()
                #print(pred) # to numpy'''
                '''
                print(type(temp_uids)) 
                print(type(files))
                print(type(zero_pred))
                print(type(one_pred))
                print(type(pred))
                print(type(labels))
                '''
                '''#<class 'numpy.ndarray'> 
                #<class 'list'>
                #<class 'numpy.ndarray'> 
                #<class 'numpy.ndarray'> 
                #<class 'torch.Tensor'> 
                # pip install torch_tb_profiler
                # different models
                # trace data ingestion, debuggertool
                # d = {"uid": temp_uids.flatten(),"file":files, "pred": pred.flatten(),"label": labels.flatten()} # tutor'''

        predictions = predictions.sort_values(["file", "overall frame number"])
        predictions = predictions.reset_index(drop=True)
        tim = {"temporal_frame_start_times": st_time}
        time_secs = pd.DataFrame(tim)
        print('Finished Testing')        
        return predictions, time_secs

    """
    Function: test_load_step
    Input: test_dataset and batch_size, 
    output: test_out are the predictions the model made.
    purpose: Allow us to load older model weights and evaluate predictions
    """
    def test_load_step(self, test_dataset, hop_length, sr, batch_size=64, model_weights=None, window_size=2):
        if model_weights != None:
            self.model.load_state_dict(torch.load(model_weights))
            
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_out = self.testing_step(test_data_loader,hop_length,sr,window_size)
        return test_out

    def load_weights(self, model_weights):
        self.model.load_state_dict(torch.load(model_weights))
   
    def test_path(self, wav_path, n_mels):
        test_spectrogram =  wav2spc(wav_path, n_mels=n_mels)
        print(test_spectrogram.shape)
        wav_data = CustomAudioDataset( test_spectrogram, [0]*test_spectrogram.shape[1], wav_path)
        test_data_loader = DataLoader(wav_data, batch_size=1)
        test_out = self.test_a_file(test_data_loader)
        return test_out

    def test_a_file(self, test_loader):
        predictions = pd.DataFrame()
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels, uids = data
                print(inputs)
                print(labels)
                print(uids)
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[0], inputs.shape[1])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                output = self.model(inputs)
                pred = torch.argmax(output, dim=1)
                d = {"uid": uids, "pred": pred.flatten(), "label": labels.flatten()}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)
        return predictions

def increase_uid(row,i):
    species = row['uid']
    spec = species.split('_')
    uid_num = int(spec[0]) + (i*86)
    spec[0] = str(uid_num)
    spec_to = '_'.join(spec)
    return spec_to

def increase_overall_frame_number(row,i):
    f_num = row['overall frame number']
    f_num = f_num+ (i*86)
    return f_num

def increase_time(row,i):
    time = row['temporal_frame_start_times']
    time += (i*2)
    return time

def prediction_fix(new_preds, number_time_bins):
    nips_fix = []
    #print(new_preds.shape)
    #print(pd.unique(new_preds["file"]))
    for fil in pd.unique(new_preds["file"]):
        windows = new_preds[new_preds["file"] == fil].shape[0]//number_time_bins
        sorted_preds = new_preds[new_preds["file"] == fil].sort_values("overall frame number")
        #print(f'{windows} windows')
        for i in range(windows):
            #print(f'Window {i + 1} starts at index {i * 86} and ends at index {(i+1)*86}')
            curr_win = sorted_preds.iloc[i * 86:(i+1)*86,:].copy(True)
            curr_win['uid'] = curr_win.apply(lambda row: increase_uid(row,i),axis = 1)
            curr_win['temporal_frame_start_times'] = curr_win.apply(lambda row: increase_time(row,i),axis = 1)
            #curr_win['overall frame number'] = curr_win.apply(lambda row: increase_overall_frame_number(row, i), axis=1)
            nips_fix.append(curr_win)

    nips_fix = pd.concat(nips_fix)
    return nips_fix