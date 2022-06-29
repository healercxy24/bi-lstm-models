# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataprocess import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)


#%% get dataset

dataset_name = 'FD001'
seq_len1 = 68
target_len = 5
seq_len2 = 36


dataset = get_dataset(dataset_name, seq_len1, target_len, seq_len2);
train_seq1 = dataset['lower_train_seq_tensor'].to(device) # size [16000, 50, 18] [dataset_len, seq_len, num_features]
train_label1 = dataset['lower_train_label_tensor'].to(device) # [16000] [dataset_len]
train_seq2 = dataset['upper_train_seq_tensor'].to(device)
train_label2 = dataset['upper_train_label_tensor'].to(device) 

valid_seq1 = dataset['lower_valid_seq_tensor'].to(device)
valid_label1 = dataset['lower_valid_label_tensor'].to(device)   # [4581]
valid_seq2 = dataset['upper_valid_seq_tensor'].to(device)
valid_label2 = dataset['upper_valid_label_tensor'].to(device)

test_seq1 = dataset['lower_test_seq_tensor'].to(device)   # size [13046, 50, 18]
test_label1 = dataset['lower_test_label_tensor'].to(device)
test_seq2 = dataset['upper_test_seq_tensor'].to(device) 
test_label2 = dataset['upper_test_label_tensor'].to(device)


#%% training

def train(model1, model2, criterion, optimizer1, optimizer2, batch_size):
    """

    Parameters
    ----------
    input1 : tensor -> size : (num_fea, batch_size, seq_len1)
        
    target1 : tensor -> size : (num_fea, batch_size, target_len)

    input2 : tensor -> size : (num_fea, batch_size, seq_len2 + target_len)
    
    target2 : tensor -> size : (batch_size)
    
    prediction1 : tensor -> size : (num_fea, batch_size, target_len)
    
    prediction1 : tensor -> size : (batch_size)
    
    batch_size : TYPE
        DESCRIPTION.

    Returns
    -------
    total_train_loss : TYPE
        DESCRIPTION.

    """
    model1.train() 
    model2.train()# turn on train mode

    total_train_loss = 0
    num_batches = train_seq1.shape[0] // batch_size
    
    for batch, i in enumerate(range(0, num_batches*batch_size, batch_size)):

        # compute the loss for the lower-level
        input1, target1 = get_batch1(train_seq1, train_label1, i, batch_size) 
        input1 = input1.permute(2, 1, 0).float()    # [4, 256, 68]
        target1 = target1.permute(2, 1, 0).float()  # [4, 256, 5]
        prediction1 = model1(input1)    # [4, 256, 5]
        loss1 = criterion(prediction1, target1) 
        
        input2, target2 = get_batch2(train_seq2, train_label2, i, batch_size) #target2 [256]
        input2 = input2.permute(2, 1, 0).float()  # [4, 256, 36]
        input2 = torch.cat((input2, prediction1), dim=2)   # [4, 256, 41]
        input2 = input2.permute(1, 0, 2)
        input2 = input2.reshape(batch_size, -1)  #[256, 164]
        prediction2 = model2(input2)
        loss2 = criterion(prediction2, target2) 
        
        loss = loss1*0.59 + loss2*(1-0.59)
        optimizer1.zero_grad()  
        optimizer2.zero_grad() 
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer1.step()
        optimizer1.step()
        
        
        total_train_loss += loss.item()
        
    total_train_loss /= num_batches
    
    return total_train_loss


def evaluate(model1, model2, criterion, batch_size):

    model1.eval()  # turn on evaluation mode
    model2.eval()

    total_valid_loss = 0
    num_batches = valid_seq1.shape[0] // batch_size
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, (num_batches-1)*batch_size, batch_size)):

            # compute the loss for the lower-level
            input1, target1 = get_batch1(valid_seq1, valid_label1, i, batch_size) 
            input1 = input1.permute(2, 1, 0).float()    # [4, 256, 68]
            target1 = target1.permute(2, 1, 0).float()  # [4, 256, 5]
            prediction1 = model1(input1)    # [4, 256, 5]
            loss1 = criterion(prediction1, target1) 
            
            input2, target2 = get_batch2(valid_seq2, valid_label2, i, batch_size) #target2 [256]
            input2 = input2.permute(2, 1, 0).float()  # [4, 256, 36]
            input2 = torch.cat((input2, prediction1), dim=2)   # [4, 256, 41]
            input2 = input2.permute(1, 0, 2)
            input2 = input2.reshape(batch_size, -1)  #[256, 164]
            prediction2 = model2(input2)
            loss2 = criterion(prediction2, target2)  

            loss = loss1*(1-0.59) + loss2*0.59            
            
            total_valid_loss += loss.item()
            
        total_valid_loss = total_valid_loss / num_batches
        

    return total_valid_loss


def test(model1, model2, criterion, batch_size):

    model1.eval()  # turn on evaluation mode
    model2.eval()

    total_test_loss = 0
    num_batches = test_seq1.shape[0] // batch_size
      
    
    with torch.no_grad():

        for batch, i in enumerate(range(0, num_batches*batch_size, batch_size)):
            
            # compute the loss for the lower-level
            input1, target1 = get_batch1(test_seq1, test_label1, i, batch_size) 
            input1 = input1.permute(2, 1, 0).float()    # [4, 256, 68]
            target1 = target1.permute(2, 1, 0).float()  # [4, 256, 5]
            prediction1 = model1(input1)    # [4, 256, 5]
            loss1 = criterion(prediction1, target1) 
            
            input2, target2 = get_batch2(train_seq2, train_label2, i, batch_size) #target2 [256]
            input2 = input2.permute(2, 1, 0).float()  # [4, 256, 36]
            input2 = torch.cat((input2, prediction1), dim=2)   # [4, 256, 41]
            input2 = input2.permute(1, 0, 2)
            input2 = input2.reshape(batch_size, -1)  #[256, 164]
            prediction2 = model2(input2)
            loss2 = criterion(prediction2, target2)  

            loss = loss1*0.59 + loss2*(1-0.59)                
            
            total_test_loss += loss.item()
            
        total_test_loss = total_test_loss / num_batches
        

    return total_test_loss
    
    
#%% running

import time
import optuna
import plotly
import operator

def objective(trial):
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1) 
    #learning_rate = 2.0
    nlayers = trial.suggest_int('nlayers', 2, 6)
    dropout = trial.suggest_loguniform('dropout', 0.001, 0.5)
    hidden_size = trial.suggest_int('hidden_size', 50, 600, 50)    
    batch_size = trial.suggest_int('batch_size', 256, 256)
    num_epochs = 100
    
    
    model1 = LowerLSTM(68, 28, 6).to(device)    
    model2 = UpperLSTM(41*4, hidden_size, nlayers, dropout).to(device) 
    criterion = nn.MSELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 1.0, gamma=0.95)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 1.0, gamma=0.95)

    #best_result = study.best_value
    best_result = float('inf')
    
    
    
    trainloss = []
    validloss = []
    
    for epoch in range(1, num_epochs + 1):
        
        epoch_start_time = time.time()

        train_loss = train(model1, model2, criterion, optimizer1, optimizer2, batch_size)
        valid_loss = evaluate(model1, model2, criterion, batch_size)
        test_loss = test(model1, model2, criterion, batch_size)
        
        trainloss.append(train_loss)
        validloss.append(valid_loss)
        

        
        if epoch % 1 == 0:
            
            elapsed = time.time() - epoch_start_time
            
            print('-' * 89)

            print(f'| end of epoch: {epoch:3d} | time: {elapsed:5.2f}s | ')
            print(f' | train loss: {train_loss:5.2f} ')
            print(f' | valid loss: {valid_loss:5.2f} ')
            print(f' | test loss: {test_loss:5.2f} ')
            #print(optimizer.state_dict()['param_groups'][0]['lr'])

            print('-' * 89)
            
            
            # save the best result with the smallest test loss
            store_addr = 'temp_model_' + dataset_name + "_" + '.pk1' 
            if test_loss < best_result:
                best_result = test_loss            
                torch.save(model2, store_addr)
    
    #print(f' | test loss: {test_loss:5.2f} ')
    #torch.save(model1, 'temp_model.pk1')
    
    scheduler1.step()
    scheduler2.step()
    
    # plot
    plt.plot(range(num_epochs), trainloss, label='train loss')
    plt.plot(range(num_epochs), validloss, label='valid loss')
    plt.legend()
    plt.show()

    return best_result


study_store_addr_li = "sqlite:///%s_fea_li.db" % (dataset_name)
study_store_addr_HI = "sqlite:///%s_fea_HI.db" % (dataset_name)
#study = optuna.create_study(study_name='linearpredict_optim', direction="minimize", storage = study_store_addr_li, load_if_exists=True)
study = optuna.create_study(study_name='HIpredict_optim_'+ dataset_name, direction="minimize", storage = study_store_addr_HI, load_if_exists=True)  
study.optimize(objective, n_trials=1)


print('study.best_params:', study.best_params)
print('study.best_value:', study.best_value)