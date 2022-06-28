# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 18:37:08 2022

@author: cxy
"""

import pandas as pd
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

#%% add the RUL column
def add_RUL(dataset):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    dataset = dataset.merge(rul, on=['id'], how='left')
    # add a column 'RUL'
    dataset['RUL'] = dataset['max'] - dataset['cycle']
    # drop 'max' column
    dataset.drop('max', axis=1, inplace=True)
    
    return dataset


#%% training set processing

def trainset_process(train_path):
    """
    Parameters:
        train_path
    return:
        test_df ->DataFrame; [13096, 20]
        in total 24 features(including 3 setting operations and 21 sensor datas)
    """
    
    train_df = pd.read_csv(train_path, sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True) # 26,27 columns are NULL
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
    train_df = train_df.sort_values(['id', 'cycle'])
    
    
    # minmaxscaler
    title = train_df.iloc[:, 0:2]
    data = train_df
    #data = train_df.iloc[:, 2:]
    
    data_norm = (data - data.min())/(data.max() - data.min()) 
    data_norm = data_norm.fillna(0) # replace all the NaN with 0
    title = add_RUL(title)
    
    #train_df = pd.concat([title, data_norm], axis=1)
    
    # group the training set with unit
    # train_set = train_df.groupby(by="id")
    
    return title, data_norm

#train_path = './CMAPSSDataNASA/train_FD001.txt'
#train_df, title, data = trainset_process(train_path)
#%% testing set processing

def testset_process(test_path, truth_path):
    """
    input:
        test_path;
    return:
        test_df -> DataFrame [13096, 20];
        in total 24 features(including 3 setting operations and 21 sensor datas)

    """
    
    test_df = pd.read_csv(test_path, sep=" ", header=None)
    # drop 26, 27 because they are NaN
    # drop setting3, s1, s5, s10, s16, s18, s19 in total 7 features because they remian unchanged
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 
                    's16', 's17', 's18', 's19', 's20', 's21']
    # First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
    test_df = test_df.sort_values(['id', 'cycle'])
    
    
    # truth RUL of test set
    truth_df = pd.read_csv(truth_path, sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    
    
    # minmaxscaler
    title = test_df.iloc[:, 0:2]
    data = test_df
    #data = test_df.iloc[:, 2:]
    data_norm = (data - data.min())/(data.max() - data.min())
    data_norm = data_norm.fillna(0) # replace all the NaN with 0
    
    # test_df = pd.concat([title, data_norm], axis=1)
    
    # generate rul for each cycle
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    # rul.columns = ['id', 'more'] # [100,2] the rul for each id
    truth_df.columns = ['rul']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = truth_df['rul'] + rul['cycle']
    truth_df.drop('rul', axis=1, inplace=True)
    
    # generate 'RUL' for test set
    """
    title [id, cycle]
    truth_df [id, max, RUL]
    """
    title['RUL'] = 0
    for i in range(0, title.shape[0]):
        id = title.iloc[i, 0]
        title.loc[i, 'RUL'] = truth_df.iloc[id-1, 1] - title.iloc[i, 1]
    
    #test_df = pd.concat([title, data_norm], axis=1)

    
    return title, data_norm


test_path = './CMAPSSDataNASA/test_FD001.txt'
truth_path = './CMAPSSDataNASA/RUL_FD001.txt'
#test_data, test_title, test_datanorm = testset_process(test_path, truth_path)
#%% PCA processing with numpy
import numpy as np


def pca(input_data, M): # M is the components you want
    """
    input:
        X: ->narray of float64;
        M: ->int: the number of PCA features
    return:
        data: ->narrya of float64;
    """
    
    input_data = input_data.values
    #mean of each feature
    n_samples, n_features = input_data.shape
    mean=np.array([np.mean(input_data[:,i]) for i in range(n_features)])
    #normalization
    norm_input = input_data-mean
    #scatter matrix
    scatter_matrix=np.dot(np.transpose(norm_input),norm_input)
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    [x[0] for x in eig_pairs].sort(reverse=True)
    # select the top k eig_vec
    feature=np.array([ele[1] for ele in eig_pairs[:M]])
    #get new data
    data=np.dot(norm_input,np.transpose(feature))
    # data=pd.DataFrame(data)
    # data.columns=['s1','s2','s3','s4']
    
    return data


#pca_traindata = pca(train_df, M=4)
#pca_testdata = pca(test_datanorm, 4)
#%% slicing window to get sequences for lower level

# generate the sequence according to the time step
def gen_sequence(data, seq_length):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length-5), range(seq_length, num_elements-5)):
        yield data[start:stop, :]

def gen_labels(data, seq_length): # predicted sequence
    
    num_elements = data.shape[0]
    for start, stop in zip(range(67, num_elements - seq_length), range(seq_length+67, num_elements)):
        yield data[start:stop, :]


# transform the raw data to sequences
def sliding_window_selection(input, title, J, K):
    """
    inputs:
        input: ->DataFrame:  given series data;
        T: ->int: time step = J = 68;
    return:
        seq_tensor: ->Tensor [];
    """
    
    
    seq_gen = list(gen_sequence(input, seq_length=J))

    seq_array = np.array(seq_gen)
    seq_tensor = torch.tensor(seq_array)
    # seq_tensor = seq_tensor.view(seq_tensor.shape[0], -1)
    seq_tensor = seq_tensor.float().to(device)
    
    # generate sequence for labels
    label_gen = list(gen_labels(input, seq_length=K))
    label_array = np.array(label_gen)
    label_tensor = torch.tensor(label_array)
    # label_tensor = label_tensor.view(label_tensor.shape[0], -1)
    label_tensor = label_tensor.float().to(device)
    
    return seq_tensor, label_tensor


#lower_test_seq_tensor, lower_test_label_tensor = sliding_window_selection(pca_testdata, test_title, 68, 5)
#%% sliding window for upper level

def get_seq_upper(input, J, L):
    
    num_elements = input.shape[0]
    for start, stop in zip(range(J - L, num_elements - L -5), range(J, num_elements-5)):
        yield input[start:stop, :]


def sliding_window_upper_level(input, title, J, L):
    
    seq_gen = list(get_seq_upper(input, J, L))
    seq_array = np.array(seq_gen)
    seq_tensor = torch.tensor(seq_array)
    seq_tensor = seq_tensor.float().to(device)
    
    # generate labels for upper level
    label_seq = title['RUL']
    label_seq.drop(label_seq.index[-6:], inplace=True)
    label_seq.drop(label_seq.index[0:J-1], inplace=True)
    label_array = np.array(label_seq)
    label_tensor = torch.tensor(label_array).to(device)

    return seq_tensor, label_tensor
