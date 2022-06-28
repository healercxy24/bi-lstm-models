# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:03:47 2022

@author: cxy
"""
# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import optuna
from data_process_original import *
from MyModel import *


torch.manual_seed(1)
torch.cuda.manual_seed(2)
 
 
device = torch.device('cuda')


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
num_epochs = 200
batch_size = 256

#learning_rate = 8.1E-3
loss_lambda = 0.59 # loss function weighting factor

df = pd.DataFrame(np.arange(12))
print(df)
df.drop(df.index[0:5], inplace=True)
print(df)