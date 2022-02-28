import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import librosa
from torch import cuda

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
    def __init__(self, num_classes, input_shape, device = torch.cuda.device('cuda'), epochs = 1,batchsize = 32, binary=False, criterion=None, optimizer=None):
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
                               rnn_dropout=0.,
                               num_layers=1
                               )
        self.device = torch.cuda.current_device()
        self.model.to(self.device)
        self.binary = binary
        self.window_size = input_shape[-1]
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
    def train_pipeline(self, train_dataset, val_dataset=None, test_dataset=None, lr=.005, batch_size=64,
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
        test_out = []

        if test_dataset != None:
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory = True,shuffle=True)

            test_out = self.testing_step(test_data_loader)

        if save_me: # save to temp?
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            # torch.save(self.model.state_dict(), os.path.join(outdir,f"model_weights-{date_str}.h5"))
            torch.save(self.model.state_dict(), f"model_weights-{date_str}.h5")
            
        # cwd = os.getcwd() # might have to be inherited from runfile
        # os.chdir(outdir) 
        self.print_results(history)  # save to out, saves to wd. works on cpu. get it to work on GPU
        #os.chdir(cwd)
        return history, test_out, start_time, end_time, date_str



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
            
            for i, data in enumerate(train_loader,0):
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
                
                # print(inputs.shape) # CUDA error: out of memory
                
                
                # 


                '''# print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
                # print(f'device in training step {torch.cuda.get_device_name(self.device)}')'''
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                print('Training Step')
                '''# print(f'labels shape in train {labels.shape}')# tutor

                # print(f'are inputs from training step in GPU? {inputs.is_cuda}')
                # print(f'are labels from training step in GPU? {labels.is_cuda}')
                # print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
                # print(f'shape inputs in train {inputs.shape}') # tutor
                # print(f'EPOCH {e}')'''



                '''#print(inputs.type())

                # print(labels.type())
                #print(inputs)'''
                
                
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

                # get statistics
                # running_loss = loss.item() # tutor
                running_loss += loss.item()

                #argmax or max??? arg returns the indices and max just the element
                # output = torch.max(output, dim=1)[1]#.squeeze() #.cpu().detach().numpy() #returns max index #OG
                output = torch.max(output, dim=1)[1]#.squeeze()
                # print(f"shape in train {output.shape}") # tutor
                # return 
                # print(f'argmax of output {output}')
                # return
                #convert output values to binary.
                correct += (output == labels).float().sum().cpu().detach().numpy()
                # print(correct)
                '''# return

                # print(f'running loss type {type(running_loss)}')
                # print(f'correct type {type(correct)}')

                # causing issue FLAG
                # for j in range(len(labels)):
                    # edit_distance += syllable_edit_distance(output[j], labels[j])'''

                # print update Improve this to make it better Maybe a global counter
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss ))

            history["loss"].append(running_loss)
            history["acc"].append(100 * correct / (len(train_loader.dataset) * self.window_size))
            # history["edit_distance"].append(edit_distance / (len(train_loader.dataset) * self.window_size))
            if val_loader != None:
                self.validation_step(val_loader, history)
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

                

                '''# inputs, labels = inputs.to(self.device), labels.to(self.device)

                # print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
                # print(f'device in training step {torch.cuda.get_device_name(self.device)}')'''
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                '''# print(f'are validation inputs from training step in GPU? {inputs.is_cuda}')
                # print(f'are validation labels from training step in GPU? {labels.is_cuda}')
                # print(f'using device {torch.cuda.get_device_name(torch.cuda.current_device())}')
                # print(f'input shape in val {inputs.shape}') # tutor'''
                print('Validation Step')

                output = self.model(inputs)
                '''# output = output.reshape(labels.shape[0], labels.shape[1]) #tutor
                #if self.binary:
                #    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))'''
                loss = self.criterion(output, labels)
                '''# check the outputs and labels here too.
                # get statistics
                # val_loss = loss.item() # tutor'''
                val_loss += loss.item() #curr

                '''
                1) You should check that the model is set to model.train() before training
                2) check that the running_loss is not being accumulated
                For valid/train
                3) train for much longer epochs
                4) scan different learning rates
                5) print out the output prediction after self.model(inputs) and compare it with the labels
                While it's training
                Print a comparison after each 100 epochs
                Between 1 sample
                So -> print(output[0])
                And -> print(labels([0[)
                '''

                #argmax or max??? arg returns the indices and max just the element

                # output = torch.max(output, dim=1)[1]
                output = torch.max(output, dim=1)[1]#.squeeze()
                # print(f'output shape in val {output}')
                
                '''
                #convert outputs to binary 0 less than .5
                '''
                # print a comparison after 100 epochs a sample, output of zero and labels zero
                val_correct += (output == labels).float().sum().cpu().detach().numpy()

                '''# for j in range(len(labels)):
                #     val_edit_distance += syllable_edit_distance(output[j], labels[j])'''

            history["val_loss"].append(val_loss)
            history["val_acc"].append(100 * val_correct / (len(val_loader.dataset) * self.window_size))
            # history["val_edit_distance"].append(val_edit_distance / (len(val_loader.dataset) * self.window_size))
            if history["val_acc"][-1] > history["best_weights"]:
                torch.save(self.model.state_dict(), "best_model_weights.h5")
                history["best_weights"] = history["val_acc"][-1]



    """
    Function: testing_step
    Input: test_loader is the test dataset
    output: predictins that the model made
    purpose: Evaluate our model on a test set
    """
    def testing_step(self, test_loader, hop_length, sr):
        predictions = pd.DataFrame()
        self.model.eval()

        st_time = []
        for i in range(86): # will change to be more general, does it only for one trainfile?
            st_time.append(get_time(i, hop_length, sr))

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
                print('Testing Step')

                output = self.model(inputs) # what is this output look like?, specify batch size here?
                '''# output = output.reshape(labels.shape[0], labels.shape[1]) # tutor
                # print(f'Testing step output shape {output.shape}') # 100, value (torch.Size([41,2,216]))
                # return 
                #print(output)
                #print(type(labels))'''
                temp_uids = []
                files = []



                if self.binary: # weakly labeled
                    print("binary")
                    labels = np.array([[x] * output.shape[-1] for x in labels]) #removed torch.from_numpy()
                    #print(labels)
                    temp_uids = np.array([[x] * output.shape[-1] for x in uids])
                    files.append(u)
                else:  # in the case of strongly labeled data
                    print('else_statement')
                    for u in uids:
                        for j in range(output.shape[-1]):
                             temp_uids.append(str(j) + "_" + u)
                             files.append(u)
                    temp_uids = np.array(temp_uids)


                    
                #print(type(labels))
                #print(labels)
                labels = labels.cpu().detach().numpy()

                zero_pred = output[:, 0, :].cpu().detach().numpy()# curr
                one_pred = output[:, 1, :].cpu().detach().numpy()# curr

                # Are we getting the correct prediction? in the sense that its the predictions calculated not something else?
                # pred = torch.max(output, dim=1)[1].squeeze().cpu().detach().numpy() # causing problems
                pred = torch.max(output, dim=1)[1].cpu().detach().numpy()

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
                d = {"uid": temp_uids.flatten(),"file":files,"zero_pred": zero_pred.flatten(), "one_pred": one_pred.flatten(), "pred": pred.flatten(),"label": labels.flatten()}

                new_preds = pd.DataFrame(d)

                predictions = predictions.append(new_preds)

                tim = {"temporal_frame_start_times": st_time}
                time_secs = pd.DataFrame(tim)

                nu_time = pd.concat([time_secs]*425, ignore_index=True)

                extracted_col = nu_time["temporal_frame_start_times"]
                
                predictions_timed = predictions.join(extracted_col)

                #process the predictions here?

        print('Finished Testing')
        return predictions_timed, time_secs

    """
    Function: test_load_step
    Input: test_dataset and batch_size, 
    output: test_out are the predictions the model made.
    purpose: Allow us to load older model weights and evaluate predictions
    """
    def test_load_step(self, test_dataset, hop_length, sr, batch_size=64,model_weights=None):
        if model_weights != None:
            self.model.load_state_dict(torch.load(model_weights))
            
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_out = self.testing_step(test_data_loader,hop_length,sr)
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
