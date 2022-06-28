# -*- coding:utf-8 -*-
"""

C-MAPSS dataset cnn
RUL prediction

author: cxy
time: 2022/03/19
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


torch.manual_seed(2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% use pandas to process csv files  -> DataFrame

# training set
train_df = pd.read_csv('./CMAPSSDataNASA/train_FD001.txt', sep=" ", header=None)  # train_dr.shape=(20631, 28)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # drop column 26 and 27, replace the tensor with the new one
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                    's18', 's19', 's20', 's21']

# First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
train_df = train_df.sort_values(['id', 'cycle'])


# testing set
# [13096, 26]
test_df = pd.read_csv('./CMAPSSDataNASA/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                   's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                   's18', 's19', 's20', 's21']

# real outputs of the testing set
# [100, 1]
truth_df = pd.read_csv('./CMAPSSDataNASA/RUL_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)



#%% Training Data Labeling - generate column 'RUL'

# add a new column 'id' : index that maximun value of 'cycle' in each group
# reset_index(): reset the index as the original form
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

# merge rul to train_df according to 'id' -> 'max' column as the last column in train_df
train_df = train_df.merge(rul, on=['id'], how='left')
# add a column 'RUL'
train_df['RUL'] = train_df['max'] - train_df['cycle']
# drop 'max' column
train_df.drop('max', axis=1, inplace=True)


"""MinMax normalization train"""
# copy 'cycle' to a new column 'cycle_norm'
train_df['cycle_norm'] = train_df['cycle']

# drop 'id', 'cycle', 'RUL' columns
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])
# scale for the left columns
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
# add the dropped columns to scaled columns
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
# restore the original index
train_df = join_df.reindex(columns=train_df.columns)


#%% process the test data and truth data

"""MinMax normalization test"""
# similar to the previous operations without 'RUL' column
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)


"""generate column max for test data"""
# the first column is 'id'. The second column is the maximum 'cycle' value with the same id
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()

rul.columns = ['id', 'max']
truth_df.columns = ['more']
# add 'id' column to truth_df, value = index of truth_df + 1
truth_df['id'] = truth_df.index + 1

# truth_df['max']: real maximum lifetime for the test set
truth_df['max'] = rul['max'] + truth_df['more']
# remove 'more'
truth_df.drop('more', axis=1, inplace=True)


"""generate RUL for test data"""
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)


"""
print results:

test_df(13096, 28)

   id  cycle  setting1  setting2  ...       s20       s21  cycle_norm  RUL
0   1      1  0.632184  0.750000  ...  0.558140  0.661834     0.00000  142
1   1      2  0.344828  0.250000  ...  0.682171  0.686827     0.00277  141
2   1      3  0.517241  0.583333  ...  0.728682  0.721348     0.00554  140
3   1      4  0.741379  0.500000  ...  0.666667  0.662110     0.00831  139
...
"""



#%% get a sequence for each id

# pick a large window size of 50 cycles
sequence_length = 50


def gen_sequence(id_df, seq_length, seq_cols):

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


"""pick the feature columns"""
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)
'''
sequence_cols=['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
'''
# gen_sequence for id == 1
val = list(gen_sequence(train_df[train_df['id'] == 1], sequence_length, sequence_cols))
val_array = np.array(val)  # val_array.shape=(142, 50, 25)  142=192-50


'''
sequence_length= 50
sequence_cols= ['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 
's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
sequence_cols have 25 columns

each sequence has a shape of [50, 25]

'''


# convert the training set to a sequence for each id number -> 100 sequences
seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

# sequence -> np array
# There are in total 100 groups of data in the training set，and each will lose 'window_size' number of datas
# 20631-100*50 = 15631
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)  # seq_array.shape=(15631, 50, 25)
seq_tensor = torch.tensor(seq_array)
seq_tensor = seq_tensor.view(15631, 50, 25).to(device)


#%%generate labels for training set

def gen_labels(id_df, seq_length, label):

    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
             for id in train_df['id'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)  # label_array.shape=(15631, 1)
label_scale = (label_array-np.min(label_array))/(np.max(label_array)-np.min(label_array))

label_tensor = torch.tensor(label_scale)
label_tensor = label_tensor.view(-1)
label_tensor = label_tensor.to(device)

"""
training set:
    seq_tensor: [15631,50,25] -> inputs
    label_tensor: [15631,1] -> targets
"""

num_sample = len(label_array)

input_size = seq_array.shape[2]
hidden_size = 4
num_layers = 2

train_split = 0.8
valid_split = 0.1


# training and evaluation-------------------------------------------------------------------------------------

import time


def train() -> None:

    model.train()  # turn on train mode
    start_time = time.time()

    # src_mask = generate_square_subsequent_mask(bptt).to(device)

    train_loss = 0
        
    for (inputs, targets) in train_loader:
        predictions = model(inputs)  # predicted results
        loss = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()  # compute the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        train_loss += loss.item()            
    
    return train_loss
 

def evaluate() -> float:

    model.eval()  # turn on evaluation mode

    valid_loss = 0.
    with torch.no_grad():

        for (inputs, targets) in valid_loader:
            output = model(inputs)
            # output_flat = output.view(-1, 1).squeeze()
            loss = criterion(output, targets)
            valid_loss += loss.item()
        print('valid_loss: ', valid_loss)

    return valid_loss



#%% main train loop definition and start training

import copy
 

def fit(num_epochs, learning_rate):

    model = Model(input_size, hidden_size).to(device)

   

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

   

    best_val_loss = float('inf')

   

    best_model = None

   

    for epoch in range(1, num_epochs + 1):

        epoch_start_time = time.time()

        train()

        val_loss = evaluate()

        # val_ppl = math.exp(val_loss)

        elapsed = time.time() - epoch_start_time

        print('-' * 89)

        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '

              f'valid loss {val_loss:5.2f} ')

        print('-' * 89)

   

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model = copy.deepcopy(model)

   

        scheduler.step()

    return best_val_loss, best_model

 
(best_val_loss, best_model) = fit(num_epochs, learning_rate)
print('best_model:', best_model)
print('best_val_loss: ', best_val_loss)



#%% hyper parameter optimization

import optuna

 

def objective(trial):


    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

   

    best_result, _ = fit(30, learning_rate)

   

    return best_result

   

study = optuna.create_study()  # Create a new study.

study.optimize(objective, n_trials=30)

print('studz.best_params:', study.best_params)