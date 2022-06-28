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

#%% dataset paths and hyperparameters


# convert the csv files to DataFrame
train_path = './CMAPSSDataNASA/train_FD001.txt'
test_path = './CMAPSSDataNASA/test_FD001.txt'
truth_path = './CMAPSSDataNASA/RUL_FD001.txt'


lower_input_size = 68
lower_hidden_size = 28

upper_input_size = 41*4
upper_hidden_size = 36

M = 4 # features selected by PCA
J = 68 # time step for lowr-level model
L = 36 # time step for upper-level model
K = 5 # time step for the predicted results from lower-level model

lstm_num_layers = 2
num_epochs = 50
batch_size = 256

#learning_rate = 
loss_lambda = 0.59 # loss function weighting factor



#%% data for lower level NN


# get training data
train_title, train_datanorm = trainset_process(train_path)
pca_data = pca(train_datanorm, M)


# split into train set and valid set
pca_traindata =pca_data[0:16000, :]
pca_validdata = pca_data[16000:, :]
training_title = train_title.iloc[0:16000, :]
valid_title = train_title.iloc[16000:, :]

lower_valid_seq_tensor, lower_valid_label_tensor = sliding_window_selection(pca_validdata, valid_title, J, K)
lower_valid_seq_tensor = lower_valid_seq_tensor.permute(0, 2, 1).to(device)
lower_valid_label_tensor = lower_valid_label_tensor.permute(0, 2, 1).to(device)
# valid finish


lower_seq_tensor, lower_label_tensor = sliding_window_selection(pca_traindata, training_title, J, K)
lower_seq_tensor = lower_seq_tensor.permute(0, 2, 1).to(device)
lower_label_tensor = lower_label_tensor.permute(0, 2, 1).to(device)


"""
pca_traindata : (16000, 4)
lower_seq_tensor : torch.Size([15928, 4, 68])
lower_label_tensor : torch.Size([15928, 4, 5])

pca_validdata : (4631, 4)
lower_valid_seq_tensor : torch.Size([4559, 4, 68])
lower_valid_label_tensor : torch.Size([4559, 4, 5])
"""

# get testing data
test_title, test_datanorm = testset_process(test_path, truth_path)
pca_testdata = pca(test_datanorm, M)

lower_test_seq_tensor, lower_test_label_tensor = sliding_window_selection(pca_testdata, test_title, J, K)
lower_test_seq_tensor = lower_test_seq_tensor.permute(0, 2, 1).to(device)
lower_test_label_tensor = lower_test_label_tensor.permute(0, 2, 1).to(device)


#%% data for upper level

# the same with the lower level until the PCA processing
upper_seq_tensor, upper_label_tensor = sliding_window_upper_level(pca_traindata, training_title, J=68, L=36)
upper_seq_tensor = upper_seq_tensor.permute(0, 2, 1)
upper_label_tensor = upper_label_tensor.view(-1, 1)



# valid set
upper_valid_seq_tensor, upper_valid_label_tensor = sliding_window_upper_level(pca_validdata, valid_title, J=68, L=36)
upper_valid_seq_tensor = upper_valid_seq_tensor.permute(0, 2, 1)
upper_valid_label_tensor = upper_valid_label_tensor.view(-1, 1)
# valid finish



# get test data for upper-level model
upper_test_seq_tensor, upper_test_label_tensor = sliding_window_upper_level(pca_testdata, test_title, J, L)
upper_test_seq_tensor = upper_test_seq_tensor.permute(0, 2, 1).to(device)
upper_test_label_tensor = upper_test_label_tensor.view(-1, 1).to(device)

"""
upper_seq_tensor : torch.Size([15928, 4, 36])
upper_label_tensor : torch.Size([15928, 1])
upper_valid_seq_tensor : torch.Size([4559, 4, 36])
upper_valid_label_tensor : torch.Size([4559, 1])

"""           
#%% divide into batches

def get_seq_batch(data_source, i):

    data = data_source[i:i+batch_size,:]

    # data = data.view(-1,50,6).permute(1,0,2).contiguous()

    return data


def get_label_batch(truth_source, i):
    
    target = truth_source[i:i+batch_size,:]
    
    return target


def get_RUL_batch(truth_source, i):
    
    target = truth_source[i:i+batch_size].float()
    
    return target

    
#%% train and evaluate methods

def train(LowerModel, UpperModel, optimizer1, optimizer2):

    LowerModel.train()  # turn on train mode
    UpperModel.train()

    total_train_loss = 0
    num_batches = lower_seq_tensor.shape[0] // batch_size

    
    for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):
        # compute the loss for the lower-level
        lower_inputs = get_seq_batch(lower_seq_tensor, i) 
        lower_targets = get_label_batch(lower_label_tensor, i)

        lower_predictions = LowerModel(lower_inputs).to(device)  # predicted results
        lower_loss = criterion(lower_predictions, lower_targets)
        
        # compute the loss for the upper-level
        upper_inputs = get_seq_batch(upper_seq_tensor, i)
        upper_targets = get_RUL_batch(upper_label_tensor, i)
        
        # merge the lower<-predictions with upper_inputs in the features dim
        #lower_predictions = lower_predictions.view(-1, 4, 5)
        upper_inputs = torch.cat((upper_inputs, lower_predictions), 2)
        upper_inputs = upper_inputs.view(upper_inputs.shape[0], 1, -1)
        
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
        
    total_train_loss /= num_batches
    
    return total_train_loss
 

def evaluate(LowerModel, UpperModel):

    LowerModel.eval()  # turn on evaluation mode
    UpperModel.eval()

    total_valid_loss = 0
    num_batches = lower_valid_seq_tensor.shape[0] // batch_size

    
    with torch.no_grad():

        for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):
            lower_inputs = get_seq_batch(lower_valid_seq_tensor, i) 
            lower_targets = get_label_batch(lower_valid_label_tensor, i)
            
            lower_predictions = LowerModel(lower_inputs).to(device)  # predicted results
            lower_loss = criterion(lower_predictions, lower_targets)
            
            # compute the loss for the upper-level
            upper_inputs = get_seq_batch(upper_valid_seq_tensor, i)
            upper_targets = get_RUL_batch(upper_valid_label_tensor, i)
            
            #lower_predictions = lower_predictions.view(-1, 4, 5)
            upper_inputs = torch.cat((upper_inputs, lower_predictions), 2)
            upper_inputs = upper_inputs.view(upper_inputs.shape[0], 1, -1)
            
            upper_predictions = UpperModel(upper_inputs).to(device)
            upper_loss = criterion(upper_predictions, upper_targets)
            
            # combine both the loss from two models
            total_loss = lower_loss*(1-loss_lambda) + upper_loss*loss_lambda
            
            total_valid_loss += total_loss.item()
            if batch == 0:
                total_pre = upper_predictions
            else:
                total_pre = torch.cat((total_pre, upper_predictions), 0)
            
        total_valid_loss = total_valid_loss / num_batches

    return total_valid_loss



#%%  main running

import time
import numpy as np

"""
LowerModel = LowerLSTM(lower_input_size, lower_hidden_size, lstm_num_layers).to(device)
UpperModel = UpperLSTM(upper_input_size, 90, lstm_num_layers, 450, 0.001489).to(device)

learning_rate = 9.76E-3

criterion = torch.nn.MSELoss() # mean-squared error for regression 
optimizer1 = torch.optim.Adam(LowerModel.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(UpperModel.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, 10, gamma=0.95)

train_loss_plot = []
valid_loss_plot = []

for lower_epoch in range(1, num_epochs + 1):
        
    epoch_start_time = time.time() 
        
    total_train_loss = train(LowerModel, UpperModel, optimizer1, optimizer2)
    total_valid_loss = evaluate(LowerModel, UpperModel)
    
    scheduler.step()
    
    elapsed = time.time() - epoch_start_time
    
    train_loss_plot.append(total_train_loss)
    valid_loss_plot.append(total_valid_loss)


    print('-' * 89)

    print(f'| end of epoch: {lower_epoch:3d} | time: {elapsed:5.2f}s | ')
    print(f'| total train loss {total_train_loss:5.2f}')
    print(f'| total valid loss: {total_valid_loss:5.2f}')

    print('-' * 89)
    


x = torch.arange(1, num_epochs+1)
plt.plot(x, train_loss_plot)
plt.title("epoch - train_loss")
plt.show()

plt.plot(x, valid_loss_plot)
plt.title("epoch - valid_loss")
plt.show()

y = torch.arange(0, lower_seq_tensor.shape[0])
plt.plot(y, total_pre)
plt.title('predicts')
plt.show()
"""


#%% hyperparameters tuning
import copy
 

def fit(learning_rate, hidden_size, fc_layer_size, lstm_dropout):
    
    LowerModel = LowerLSTM(lower_input_size, lower_hidden_size, lstm_num_layers).to(device)
    UpperModel = UpperLSTM(upper_input_size, hidden_size, lstm_num_layers, lstm_dropout).to(device)

   

    criterion = nn.MSELoss()

    optimizer1 = torch.optim.Adam(LowerModel.parameters(), learning_rate)
    optimizer2 = torch.optim.Adam(UpperModel.parameters(), learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, 10, gamma=0.95)

   

    best_valid_loss = float('inf')

   

    best_model = None
    

    epoch_start_time = time.time()  

    for epoch in range(1, num_epochs + 1):


        train_loss = train(LowerModel, UpperModel, optimizer1, optimizer2)
        valid_loss = evaluate(LowerModel, UpperModel)

        # val_ppl = math.exp(val_loss)
        
        if epoch % 10 == 0:
            
            elapsed = time.time() - epoch_start_time
            
            print('-' * 89)

            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '

              f'train loss {train_loss:5.2f} ')
            print(f' | valid loss {valid_loss:5.2f} ')

            print('-' * 89)

   

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            best_model = copy.deepcopy(UpperModel)

   
        scheduler.step()

    return best_valid_loss, best_model


"""
#(best_train_loss, best_model) = fit(learning_rate, num_layers)
#print('best_model:', best_model)
#print('best_train_loss: ', best_train_loss)
"""




#%% hyper parameter optimization

import optuna

 

def objective(trial):


    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    #num_layers = trial.suggest_int('num_layers', 2, 5)
    #layer_size = trial.suggest_int('layer_size', 100, 700, 50)
    hidden_size = trial.suggest_int('hidden_size', 10, 400)
    lstm_dropout = trial.suggest_loguniform('lstm_dropout', 0.001, 1)
    fc_layer_size = trial.suggest_int('fc_layer_size', 50, 500, 50)
   

    best_result, _ = fit(learning_rate,hidden_size, fc_layer_size, lstm_dropout)
    print('best_results: ', best_result)
   

    return best_result

   

study = optuna.create_study(study_name=('valid_loss_optim'))  # Create a new study.

study.optimize(objective, n_trials=30)

print('study.best_params:', study.best_params)

