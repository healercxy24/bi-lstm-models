# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 22:45:56 2022

@author: cxy
"""

import pandas as pd


train_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/train_FD001.txt'
test_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/test_FD001.txt'
truth_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/RUL_FD001.txt'


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
        train_path;
    return:
        train_df -> DataFrame [20631, 27] (24 features and id, cycle, RUL)
    """
    
    train_df = pd.read_csv(train_path, sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True) # 26,27 columns are NULL
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
                    's16', 's17', 's18', 's19', 's20', 's21']
    # First sort by the elements in the 'id' column, and when the elements in the 'id' column are the same, sort by the 'cycle' column
    train_df = train_df.sort_values(['id', 'cycle'])
    
    
    # minmaxscaler
    title = train_df.iloc[:, 0:2]
    data = train_df.iloc[:, 2:]
    data_norm = (data - data.min())/(data.max() - data.min())
    
    train_df = pd.concat([title, data_norm], axis=1)
    
    train_df = add_RUL(train_df)
    
    # group the training set with unit
    # train_set = train_df.groupby(by="id")
    
    return train_df


#%% testing set processing

def testset_process(test_path, truth_path):
    """
    param:
        test_path, truth_path;
    return:
        test_df -> DataFrame  [13096, 27] (24 features and id, cycle, RUL)

    """
    
    test_df = pd.read_csv(test_path, sep=" ", header=None)
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
    data = test_df.iloc[:, 2:]
    data_norm = (data - data.min())/(data.max() - data.min())
    
    test_df = pd.concat([title, data_norm], axis=1)
    
    # generate rul for each cycle
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    # rul.columns = ['id', 'more'] # [100,2] the rul for each id
    truth_df.columns = ['rul']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = truth_df['rul'] + rul['cycle']
    truth_df.drop('rul', axis=1, inplace=True)
    
    # generate 'RUL' for test set
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    
    # group the testing set with unit
    #test_set = test_df.groupby(by="id")
    
    return test_df


#%% PCA processing
import torch
from sklearn import datasets
import numpy as np

from sklearn.decomposition import PCA

pca = PCA(n_components=4)

input = trainset_process(train_path)
input = np.array(input)
input = torch.Tensor(input)

pca.fit(input)
print(input)

