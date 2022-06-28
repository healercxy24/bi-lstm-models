# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:10:12 2022

@author: cxy
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch


train_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/train_FD001.txt'
test_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/test_FD001.txt'
truth_path = 'C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/RUL_FD001.txt'


train_df = pd.read_csv(train_path, sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # drop column 26 and 27, replace the tensor with the new one
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                    's18', 's19', 's20', 's21']


test_df = pd.read_csv('C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                   's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                   's18', 's19', 's20', 's21']


truth_df = pd.read_csv('C:/Users/njuxc/project1/datasets/CMAPSSDataNASA/RUL_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)


for index, column in test_df.iteritems():
    x = torch.arange(0, len(test_df))
    plt.plot(x, test_df[index])
    plt.title(index)
    plt.show()