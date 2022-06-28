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

#train_data, test_data, truth_label = dataset_process(dataset_name)
dataset = get_dataset(dataset_name, seq_len);
train_seq = dataset['lower_train_seq_tensor'] # size [16000, 50, 18] [dataset_len, seq_len, num_features]
#train_seq = train_seq.view(train_seq.shape[0], -1) # [dataset_len, seq_len*num_features] [16000, 1300]
train_label = dataset['lower_train_label_tensor'] # [16000] [dataset_len]

valid_seq = dataset['lower_valid_seq_tensor']
#valid_seq = valid_seq.view(valid_seq.shape[0], -1)
valid_label = dataset['lower_valid_label_tensor']   # [4581]

test_seq = dataset['lower_test_seq_tensor']   # size [13046, 50, 18]
test_label = dataset['lower_test_label_tensor']