# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:42:32 2022

@author: njucx
"""

import torch
import torch.nn as nn
from data_process_original import *
from MyModel import *
import matplotlib.pyplot as plt

device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)

#%% dataset paths and hyperparameters


# convert the csv files to DataFrame
train_path = './CMAPSSDataNASA/train_FD001.txt'
test_path = './CMAPSSDataNASA/test_FD001.txt'
truth_path = './CMAPSSDataNASA/RUL_FD001.txt'


lower_input_size = 68
lower_hidden_size = 28

upper_input_size = 41*4
#upper_hidden_size = 36

lstm_num_layers = 2

M = 4 # features selected by PCA
J = 68 # time step for lowr-level model
L = 36 # time step for upper-level model
K = 5 # time step for the predicted results from lower-level model

#lstm_num_layers = 2
num_epochs = 1
batch_size = 256

#learning_rate = 8.1E-3
loss_lambda = 0.59 # loss function weighting factor


#%% data for lower level NN

"""
process the data with PCA and output the data_array, title_array and merged input_daya_array

training set:
    train_data : (20631, 27)
    train_title : (20631, 3)
    train_datanorm : (20631, 24)
    pca_traindata : (20631, 4)
    lower_seq_tensor : torch.Size([20558, 4, 68])
    lower_label_tensor : torch.Size([20558, 4, 5])


testing set:
    test_data : (13096, 27)
    test_title : (13096, 3)
    test_datanorm : (13096, 24)
    pca_testdata : (13096, 4)
    lower_test_seq_tensor : (13024, 4, 68)
    lower_test_label_tensor : (13024, 4, 5)
"""

    
    # get training dataset
train_title, train_datanorm = trainset_process(train_path)
pca_traindata = pca(train_datanorm, M)
    
lower_seq_tensor, lower_label_tensor = sliding_window_selection(pca_traindata, train_title, J, K)
lower_seq_tensor = lower_seq_tensor.permute(0, 2, 1).to(device)
lower_label_tensor = lower_label_tensor.permute(0, 2, 1).to(device)





# split into train&valid
seq_list = np.array(lower_seq_tensor.cpu())
label_list = np.array(lower_label_tensor.cpu())

np.random.seed(123)
temp = np.arange(0, len(seq_list))
np.random.shuffle(temp)


seq_tensor = []
label_tensor = []
for i in temp:
    seq_tensor.append(seq_list[i])
    label_tensor.append(label_list[i])

seq_tensor = np.array(seq_tensor)
label_tensor = np.array(label_tensor)
seq_tensor = torch.tensor(seq_tensor)
label_tensor = torch.tensor(label_tensor)


train_seq_tensor = seq_tensor[0:16000,:].to(device)
train_label_tensor = label_tensor[0:16000,:].to(device)
valid_seq_tensor = seq_tensor[16000:,: ].to(device)
valid_label_tensor = label_tensor[16000:,: ].to(device)



# get testing data
test_title, test_datanorm = testset_process(test_path, truth_path)
pca_testdata = pca(test_datanorm, M)

lower_test_seq_tensor, lower_test_label_tensor = sliding_window_selection(pca_testdata, test_title, J, K)
lower_test_seq_tensor = lower_test_seq_tensor.permute(0, 2, 1).to(device)
lower_test_label_tensor = lower_test_label_tensor.permute(0, 2, 1).to(device)


#%% data for upper level
"""
the input seqeuence is made of two parts:
    first: upper_seq_
    second: 
"""

# the same with the lower level until the PCA processing
upper_seq_tensor, upper_label_tensor = sliding_window_upper_level(pca_traindata, train_title, J=68, L=36)
upper_seq_tensor = upper_seq_tensor.permute(0, 2, 1).to(device)
upper_label_tensor = upper_label_tensor.view(-1).to(device)




# split into train&valid
upper_seq_list = np.array(upper_seq_tensor.cpu())
upper_label_list = np.array(upper_label_tensor.cpu())

#upper_temp = np.arange(0, len(upper_seq_list))
#np.random.shuffle(upper_temp)


upper_seq_tensor = []
upper_label_tensor = []

for i in temp:
    upper_seq_tensor.append(upper_seq_list[i])
    upper_label_tensor.append(upper_label_list[i])


upper_seq_tensor = np.array(upper_seq_tensor)
upper_label_tensor = np.array(upper_label_tensor)
upper_seq_tensor = torch.tensor(upper_seq_tensor)
upper_label_tensor = torch.tensor(upper_label_tensor)


upper_train_seq_tensor = upper_seq_tensor[0:16000,:].to(device)
upper_train_label_tensor = upper_label_tensor[0:16000].to(device)
upper_valid_seq_tensor = upper_seq_tensor[16000:,: ].to(device)
upper_valid_label_tensor = upper_label_tensor[16000: ].to(device)





# get test data for upper-level model
upper_test_seq_tensor, upper_test_label_tensor = sliding_window_upper_level(pca_testdata, test_title, J, L)
upper_test_seq_tensor = upper_test_seq_tensor.permute(0, 2, 1).to(device)
upper_test_label_tensor = upper_test_label_tensor.view(-1).to(device)


#%% divide into batches

def get_seq_batch(data_source, i):  # to generate sequences for both train and test dataset

    data = data_source[i:i+batch_size,:]

    # data = data.view(-1,50,6).permute(1,0,2).contiguous()

    return data


def get_label_batch(truth_source, i):  # generate target sequence for lower-level model
    
    target = truth_source[i:i+batch_size,:]
    
    return target


def get_RUL_batch(truth_source, i): # generate target RUL value for upper-level prediction
    
    target = truth_source[i:i+batch_size].float()
    
    return target

    
#%% train and evaluate methods


def train(LowerModel, UpperModel, criterion, optimizer1, optimizer2):

    LowerModel.train()  # turn on train mode
    UpperModel.train()

    total_train_loss = 0
    num_batches = train_seq_tensor.shape[0] // batch_size

    
    total_pre = []

    
    for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):
        # compute the loss for the lower-level
        lower_inputs = get_seq_batch(train_seq_tensor, i) 
        lower_targets = get_label_batch(train_label_tensor, i)

        lower_predictions = LowerModel(lower_inputs).to(device)  # predicted results
        lower_loss = criterion(lower_predictions, lower_targets)
        
        # compute the loss for the upper-level
        upper_inputs = get_seq_batch(upper_train_seq_tensor, i)
        upper_targets = get_RUL_batch(upper_train_label_tensor, i)
        
        # merge the lower<-predictions with upper_inputs in the features dim
        #lower_predictions = lower_predictions.view(-1, 4, 5)
        upper_inputs = torch.cat((upper_inputs, lower_predictions), 2)
        upper_inputs = upper_inputs.view(upper_inputs.shape[0], -1)
        
        # compute the results and loss for upper-level
        upper_predictions = UpperModel(upper_inputs).to(device)
        upper_loss = criterion(upper_predictions, upper_targets)
        
        
        # combine both the loss from two models
        total_loss = lower_loss*(1-loss_lambda) + upper_loss*loss_lambda
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(LowerModel.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(UpperModel.parameters(), 0.5)
        optimizer1.step()
        optimizer2.step()
        
        total_train_loss += total_loss.item()
        total_pre.append(upper_predictions)
        
    total_train_loss /= num_batches
    
    return total_train_loss, total_pre
 

def evaluate(LowerModel, UpperModel, criterion):

    LowerModel.eval()  # turn on evaluation mode
    UpperModel.eval()

    total_valid_loss = 0
    total_test_loss = 0
    num_batches = valid_seq_tensor.shape[0] // batch_size
    
    total_pre = []
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):
            lower_inputs = get_seq_batch(valid_seq_tensor, i) 
            lower_targets = get_label_batch(valid_label_tensor, i)
            
            lower_predictions = LowerModel(lower_inputs).to(device)  # predicted results
            lower_loss = criterion(lower_predictions, lower_targets)
            
            # compute the loss for the upper-level
            upper_inputs = get_seq_batch(upper_valid_seq_tensor, i)
            upper_targets = get_RUL_batch(upper_valid_label_tensor, i)
            
            #lower_predictions = lower_predictions.view(-1, 4, 5)
            upper_inputs = torch.cat((upper_inputs, lower_predictions), 2)
            upper_inputs = upper_inputs.view(upper_inputs.shape[0], -1)
            
            upper_predictions = UpperModel(upper_inputs).to(device)
            upper_loss = criterion(upper_predictions, upper_targets)
            
            # combine both the loss from two models
            total_loss = lower_loss*(1-loss_lambda) + upper_loss*loss_lambda
            
            total_valid_loss += total_loss.item()
            total_pre.append(upper_predictions)
            
        total_valid_loss = total_valid_loss / num_batches
        

    return total_valid_loss, total_pre


def test(LowerModel, UpperModel, criterion):

    LowerModel.eval()  # turn on evaluation mode
    UpperModel.eval()

    total_test_loss = 0

    num_batches = lower_test_seq_tensor.shape[0] // batch_size
    
    total_pre = []
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):
            lower_inputs = get_seq_batch(lower_test_seq_tensor, i) 
            lower_targets = get_label_batch(lower_test_label_tensor, i)
            
            lower_predictions = LowerModel(lower_inputs).to(device)  # predicted results
            lower_loss = criterion(lower_predictions, lower_targets)
            
            # compute the loss for the upper-level
            upper_inputs = get_seq_batch(upper_test_seq_tensor, i)
            upper_targets = get_RUL_batch(upper_test_label_tensor, i)
            
            #lower_predictions = lower_predictions.view(-1, 4, 5)
            upper_inputs = torch.cat((upper_inputs, lower_predictions), 2)
            upper_inputs = upper_inputs.view(upper_inputs.shape[0], -1)
            
            upper_predictions = UpperModel(upper_inputs).to(device)
            upper_loss = criterion(upper_predictions, upper_targets)
            
            # combine both the loss from two models
            total_loss = lower_loss*(1-loss_lambda) + upper_loss*loss_lambda
            
            total_test_loss += total_loss.item()
            total_pre.append(upper_predictions)
            
        total_test_loss = total_test_loss / num_batches
        

    return total_test_loss, total_pre


#%%  main running

import time
import numpy as np
import optuna



def Train_model():
    df = optuna.study.load_study('train_valid_optim', "sqlite:///20220506_1.db")
    upper_hidden_size = df.best_params['hidden_size']
    lstm_dropout = df.best_params['lstm_dropout']
    #LowerModel = LowerLSTM(lower_input_size, lower_hidden_size, num_layers=2).to(device)
    UpperModel = UpperLSTM(upper_input_size, upper_hidden_size, lstm_num_layers, lstm_dropout).to(device)
    #LowerModel = torch.load('lower_net_model_6.pk1')
    UpperModel = torch.load('upper_net_model_6.pk1')
    #print(UpperModel)
       
    learning_rate = df.best_params['learning_rate']

    criterion = torch.nn.MSELoss() # mean-squared error for regression 
    optimizer1 = torch.optim.Adam(LowerModel.parameters(), learning_rate)
    optimizer2 = torch.optim.Adam(UpperModel.parameters(), learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, gamma=0.95)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, gamma=0.95)

    train_loss_plot = []
    valid_loss_plot = []
    test_loss_plot = []
    
    for lower_epoch in range(1, num_epochs + 1):
            
        epoch_start_time = time.time() 
            
        total_train_loss, train_pre = train(LowerModel, UpperModel, criterion, optimizer1, optimizer2)
        total_valid_loss, valid_pre = evaluate(LowerModel, UpperModel, criterion)
        total_test_loss, test_pre = test(LowerModel, UpperModel, criterion)
        
        
        elapsed = time.time() - epoch_start_time
        
        train_loss_plot.append(total_train_loss)
        valid_loss_plot.append(total_valid_loss)
        test_loss_plot.append(total_test_loss)
        
        scheduler1.step()
        scheduler2.step()

        print('-' * 89)

        print(f'| end of epoch: {lower_epoch:3d} | time: {elapsed:5.2f}s | ')
        print(f'| total train loss {total_train_loss:5.2f}')
        print(f'| total valid loss: {total_valid_loss:5.2f}')
        print(f'| total test loss: {total_test_loss:5.2f}')

        print('-' * 89)
    
    torch.save(LowerModel, 'lower_net_model.pk1')   # save the whole model and params
    torch.save(UpperModel, 'upper_net_model.pk1')
    

    
    x = torch.arange(1, num_epochs+1)
    plt.plot(x, train_loss_plot)
    plt.title('epoch - training loss')
    plt.show()

    plt.plot(x, valid_loss_plot)
    plt.title('epoch - validing loss')
    plt.show()
    
    plt.plot(x, test_loss_plot)
    plt.title('epoch -testing loss')
    plt.show()
    
    
    
    # plot the RUL values for training set
    train_pre = torch.stack(train_pre)
    train_pre = train_pre.view(-1)
    train_pre = train_pre.cpu()
    label_plot = upper_label_tensor.view(-1)[0:15616].to('cpu')
    
    plt.scatter(torch.arange(1, train_pre.shape[0]+1), train_pre.detach().numpy(), label='predicted')
    plt.plot(torch.arange(1, train_pre.shape[0]+1), label_plot.detach().numpy(), label='label')
    plt.title('train_pre')
    plt.legend()
    plt.show()
    
    
    
    # plot RUL for valid set
    valid_pre = torch.stack(valid_pre)
    valid_pre = valid_pre.view(-1)
    valid_pre = valid_pre.cpu()
    label_plot = upper_valid_label_tensor.view(-1)[-4096:].to('cpu')
    
    plt.scatter(label_plot.detach().numpy(), valid_pre.detach().numpy(), label='predicted')
    plt.plot(label_plot.detach().numpy(), label_plot.detach().numpy(), 'r-', label='label')
    plt.title('valid_pre')
    plt.show()
    

run = Train_model()



#%% test the saved model and parameters


"""
LowerModel = torch.load('lower_net_model_6.pk1')
UpperModel = torch.load('upper_net_model_6.pk1')
print(UpperModel)
criterion = torch.nn.MSELoss() # mean-squared error for regression

total_valid_loss, _ = evaluate(LowerModel, UpperModel, criterion)
total_test_loss, _ = test(LowerModel, UpperModel, criterion)

print('-' * 89)

print(f'| total valid loss: {total_valid_loss:5.2f}')
print(f'| total test loss: {total_test_loss:5.2f}')

print('-' * 89)
"""



#%% hyper parameter optimization

import optuna
import plotly

import operator

"""
class EarlyStoppingCallback(object):
    # Early stopping callback for Optuna.

    def __init__(self, early_stopping_rounds: int) -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        self._operator = operator.lt
        self._score = np.inf


    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        # Do early stopping.
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop() 



def objective(trial):
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2) 
    #num_layers = trial.suggest_int('num_layers', 2, 5)
    #layer_size = trial.suggest_int('layer_size', 100, 700, 50)
    hidden_size = trial.suggest_int('hidden_size', 10, 200)
    lstm_dropout = trial.suggest_loguniform('lstm_dropout', 0.001, 0.5)
    fc_1d = trial.suggest_loguniform('fc_1_dropout', 0.001, 0.5)
    fc_2d = trial.suggest_loguniform('fc_2_dropout', 0.001, 0.5)
    
    

    #LowerModel = LowerLSTM(lower_input_size, lower_hidden_size, lstm_num_layers).to(device)
    #UpperModel = UpperLSTM(upper_input_size, hidden_size, lstm_num_layers, lstm_dropout, fc_1d, fc_2d).to(device)    
    LowerModel = torch.load('lower_net_model_6.pk1').to(device)
    UpperModel = torch.load('upper_net_model_6.pk1').to(device)
    
    
    criterion = nn.MSELoss()

    optimizer1 = torch.optim.Adam(LowerModel.parameters(), learning_rate)
    optimizer2 = torch.optim.Adam(UpperModel.parameters(), learning_rate)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, gamma=0.95)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, gamma=0.95)
    
    
    epoch_start_time = time.time()  
    best_result = float('inf')

    for epoch in range(1, num_epochs + 1):


        train_loss, _ = train(LowerModel, UpperModel, criterion, optimizer1, optimizer2)
        valid_loss, _ = evaluate(LowerModel, UpperModel, criterion)
        test_loss, _ = test(LowerModel, UpperModel, criterion)
        
        scheduler1.step()
        scheduler2.step()

        
        if epoch % 200 == 0:
            
            elapsed = time.time() - epoch_start_time
            
            print('-' * 89)

            print(f'| end of epoch: {epoch:3d} | time: {elapsed:5.2f}s | ')
            print(f' | train loss: {train_loss:5.2f} ')
            print(f' | valid loss: {valid_loss:5.2f} ')
            print(f' | test loss: {test_loss:5.2f} ')

            print('-' * 89)
    
        if valid_loss < best_result:
            best_result = valid_loss
            
            torch.save(LowerModel, 'lower_net_model_1.pk1')   # save the whole model and params
            torch.save(UpperModel, 'upper_net_model_1.pk1')

    return valid_loss


early_stopping = EarlyStoppingCallback(10) # after 10 rounds stop

study = optuna.create_study(study_name='train_valid_optim', direction="minimize", storage = "sqlite:///20220506_1.db", load_if_exists=True)  # Create a new study.

study.optimize(objective, n_trials=1, callbacks=[early_stopping])


print('study.best_params:', study.best_params)
print('study.best_value:', study.best_value)
"""
