# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% add the RUL column

def add_RUL(dataset):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    dataset = dataset.merge(rul, on=['id'], how='left')
    # add a column 'RUL'
    rul = dataset['max'] - dataset['cycle']
    # drop 'max' column
    dataset.drop('max', axis=1, inplace=True)
    

    for index, row in rul.iteritems():
        if(row > 130):
            rul[index] = 130

    
    dataset['RUL'] = rul
    
    return dataset


def add_RUL_test(dataset, truth_rul):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    rul['max'] = rul['max'] + truth_rul['RUL']
    dataset = dataset.merge(rul, on=['id'], how='left')
    # add a column 'RUL'
    rul = dataset['max'] - dataset['cycle']
    # drop 'max' column
    dataset.drop('max', axis=1, inplace=True)
    

    for index, row in rul.iteritems():
        if(row > 130):
            rul[index] = 130

    
    dataset['RUL'] = rul
    
    return dataset
    

#%% dataset processing

def dataset_process(dataset_name):
    """
    Parameters:
        dataset_name : string : 'FD001', 'FD002', 'FD003', 'FD004'
    return:
        train_set : [100,] --> [20631, 18]
        test_set : [100,] --> [20631, 18]
        in total 18 features(including : id, cycle, 2 setting operations, 14 sensor datas)
        
        train_data, train_label, test_data, test_label --> nparray
        
    """
    
    root_path = './CMAPSSDataNASA/'
    
    # set the column names
    title_names = ['id', 'cycle']
    setting_names = ['setting1', 'setting2', 'setting3']
    data_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = title_names + setting_names + data_names
    
    drop_cols = ['setting3', "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
    
    # load data from the txt file
    train_df = pd.read_csv((root_path + 'train_' + dataset_name + '.txt'), sep='\s+', header=None, names=col_names)
    # First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
    train_df = train_df.sort_values(['id', 'cycle'])
    
    test_df = pd.read_csv((root_path + 'test_' + dataset_name + '.txt'), sep='\s+', header=None, names=col_names)
    test_df = test_df.sort_values(['id', 'cycle'])
    
    rul_df = pd.read_csv((root_path + 'RUL_' + dataset_name + '.txt'), sep='\s+', header=None, names=['RUL'])
    rul_df['id'] = rul_df.index + 1
    
    train_df.drop(drop_cols, axis=1, inplace=True)
    test_df.drop(drop_cols, axis=1, inplace=True)
    
    
    '''process the train data'''
    title = train_df.iloc[:, 0:2]  #[id,cycle]
    data = train_df.iloc[:, 2:]  # [20631, 16]
    
    # minmaxscaler for data
    data_norm = (data - data.min())/(data.max() - data.min()) 
    data_norm = data_norm.fillna(0) # replace all the NaN with 0        
    # merge title & data_norm
    train_data = pd.concat([title, data_norm], axis=1).to_numpy()   # [20631, 18]    
    # add RUL col to title
    title = add_RUL(title) 
    # group the training set with 'id'
    #train_group = train_data.groupby(by="id")
    train_label = title['RUL'].to_numpy()
    
    
    '''process the test data'''
    title = test_df.iloc[:, 0:2]
    data = test_df.iloc[:, 2:]
    
    # minmaxscaler for data
    data_norm = (data - data.min())/(data.max() - data.min()) 
    data_norm = data_norm.fillna(0) # replace all the NaN with 0   
    # merge title & data_norm
    test_data = pd.concat([title, data_norm], axis=1).to_numpy()   # [13096, 18]   
    # add RUL col to title
    title = add_RUL_test(title, rul_df)  
    # group the training set with 'id'
    #test_group = test_data.groupby(by="id")
    test_label = title['RUL'].to_numpy()
    
    return train_data, train_label, test_data, test_label


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
    
    #input_data = input_data.values
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


#%% slicing window to get sequences

# generate the sequence for lower model
def gen_sequence1(data, seq_len1, target_len):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements - seq_len1 - target_len), range(seq_len1, num_elements - target_len)):
        yield data[start:stop, :]

def gen_labels1(data, seq_len1, target_len):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(seq_len1, num_elements - target_len), range(num_elements - target_len, num_elements)):
        yield data[start:stop, :]

# generate the sequence for upper model
def gen_sequence2(data, seq_len1, target_len, seq_len2):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(seq_len1 + target_len - seq_len2, num_elements - seq_len2), range(seq_len1 + target_len, num_elements)):
        yield data[start:stop, :]
        
def gen_labels2(data, seq_len1, target_len, seq_len2):
    
    num_elements = data.shape[0]
    for start, stop in zip(range(seq_len1 + target_len - seq_len2, num_elements - seq_len2), range(seq_len1 + target_len, num_elements)):
        yield data[start:stop]


#%% get processed dataset

"""
loss the number of seq_len data because of gen_sequence : 
    trainset : 20631 = 16000 + 4581 + seq_len

training set:
    train_seq_tensor : torch.Size([16000, 50, 18])
    train_label_tensor : torch.Size([16000])
    valid_seq_tensor : torch.Size([4581, 50, 18])
    valid_label_tensor : torch.Size([4581])


testing set: 
    test_seq_tensor : torch.Size([13046, 50, 18])
    test_label_tensor : torch.Size([13046])
"""

def get_dataset(dataset_name, seq_len1, target_len, seq_len2):   
    
    train_data, train_label, test_data, test_label = dataset_process(dataset_name)  
    
    '''generate sequences for trainset'''   
    # generate labels for lower model
    train_label1 = list(gen_labels1(train_data, seq_len1, target_len))
    train_label1 = np.array(train_label1)
    label_tensor1 = torch.tensor(train_label1)   # torch.Size([20581])
    label_tensor1 = label_tensor1.float().to(device)
    
    # generate seq for lower model
    train_data = pca(train_data, 4)     
    seq_array1 = list(gen_sequence1(train_data, seq_len1, target_len))  
    seq_tensor1 = torch.tensor(seq_array1)    # [20581, 50, 18]
    seq_tensor1 = seq_tensor1.float().to(device)   

    
    #split the new dataset
    train_seq_tensor1 = seq_tensor1[0:16000,:].to(device)  # [16000, 50, 18]
    train_label_tensor1 = label_tensor1[0:16000].to(device)
    valid_seq_tensor1 = seq_tensor1[16000:,: ].to(device)  # [4581, 50, 18]
    valid_label_tensor1 = label_tensor1[16000:].to(device)
    
    
    
    # generate labels for upper model
    train_label2 = list(gen_labels2(train_data, seq_len1, target_len))
    train_label2 = np.array(train_label2)
    label_tensor2 = torch.tensor(train_label2)   # torch.Size([20581])
    label_tensor2 = label_tensor2.float().to(device)
    
    # generate seq for upper model
    train_data = pca(train_data, 4)     
    seq_array2 = list(gen_sequence2(train_data, seq_len1, target_len, seq_len2))  
    seq_tensor1 = torch.tensor(seq_array1)    # [20581, 50, 18]
    seq_tensor1 = seq_tensor1.float().to(device)   

    
    #split the new dataset
    train_seq_tensor1 = seq_tensor1[0:16000,:].to(device)  # [16000, 50, 18]
    train_label_tensor1 = label_tensor1[0:16000].to(device)
    valid_seq_tensor1 = seq_tensor1[16000:,: ].to(device)  # [4581, 50, 18]
    valid_label_tensor1 = label_tensor1[16000:].to(device)
    
    
    '''process data for test dataset'''
    # generate labels
    test_label1 = list(gen_labels(test_label, seq_len))   # [13046]
    test_label = np.array(test_label)
    test_label_tensor = torch.tensor(test_label)
    test_label_tensor = test_label_tensor.float().to(device)
    
    #test_data = pca(test_data, 8)
    test_array = list(gen_sequence(test_data, seq_len))   # [13046]
    test_seq_tensor = torch.tensor(test_array)
    test_seq_tensor = test_seq_tensor.float().to(device) 
    
    
    dataset = {'lower_train_seq_tensor' : train_seq_tensor1,
               'lower_train_label_tensor' : train_label_tensor1,
               'upper_train_seq_tensor' : train_seq_tensor2,
               'upper_train_label_tensor' : train_label_tensor2,
               'lower_valid_seq_tensor' : valid_seq_tensor1,
               'lower_valid_label_tensor' : valid_label_tensor1,
               'upper_valid_seq_tensor' : valid_seq_tensor2,
               'upper_valid_label_tensor' : valid_label_tensor2,
               'lower_test_seq_tensor' : test_seq_tensor1,
               'lower_test_label_tensor' : test_label_tensor1,
               'upper_test_seq_tensor' : test_seq_tensor2,
               'upper_test_label_tensor' : test_label_tensor2
               }
    
    return dataset

#%% get batch

def get_batch(data_source, truth_source, i, batch_size):

    """

    Args:

        source: Tensor, shape [dataset_length, sequence_length, num_features]

        i: int
        
        seq_len : size of sequence
 

    Returns:

        tuple (data, target), where data has shape [sequence_length, batch_size, num_features] and

        target has shape [seq_len, batch_size]

    """

    data = data_source[i:i+batch_size, :]
    data = data.view(-1, batch_size, data.shape[2]).contiguous()

    target = truth_source[i:i+batch_size]

    return data, target
