# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


device = torch.device('cuda')
torch.manual_seed(3)
torch.cuda.manual_seed(4)


class LowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LowerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=0.12)
        self.fc_1 = nn.Linear(hidden_size, 100)
        self.fc_2 = nn.Linear(100, 50)
        self.fc = nn.Linear(50, 5)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        
        :param :
            
            x: input with size (L, N, Hin)
            h_0 : (D * num_layers, N, Hout)
            c_0 : (D * num_layers, N, Hout)
            
            L : sequence_length
            N : batch_size
            Hin : input_size
            Hcell : hidden_size
            Hout : output_size = proj_size if proj_size > 0 otherwise hidden_size
            D = 2 if ib-directional=True else = 1
            
            
        :return: 
            
            out : (L, N, D*Hout)
            h_n : (D*num_layers, N, Hout)
            c_n : (D*num_layers, N, Hcell)
            
        """
        
        h_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        
        out = self.relu(output)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc(out)

        return out


class UpperLSTM(nn.Module):           
            
    def __init__(self, input_size, hidden_size, lstm_num_layers, dropout):
        super(UpperLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = dropout
        
        self.BN = nn.BatchNorm1d(input_size) # with learnable parameters
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=lstm_num_layers, batch_first=True, dropout=self.lstm_dropout)
            
        self.fc_1 = nn.Linear(hidden_size, 140)
        self.BN_1 = nn.BatchNorm1d(140)
        self.fc_2 = nn.Linear(140, 70)
        self.BN_2 = nn.BatchNorm1d(70)
        self.fc = nn.Linear(70, 1)
        
        self.activation = nn.ReLU()
        self.Dropout = nn.Dropout(p=dropout)
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        if isinstance(self, nn.Linear):
            nn.init.xavier_uniform_(self.weight)
            nn.init.constant_(self.bias,0)

        elif isinstance(self,nn.BatchNorm1d):
            nn.init.constant_(self.weight,1)
            nn.init.constant_(self.bias,0)
    
    def forward(self, x):
        """
        
        :param x: input features [batch_size, sequence_length * input_size]
        :return: predictions results
        """

        x = self.BN(x)  
        out, _ = self.lstm(x) # [batch_size, seq_len, hidden_size]
        #x = self.BN(x)
        
        # out = self.relu(output)
        out = self.fc_1(out)
        #out = self.BN_1(out)
        out = self.Dropout(out)
        out = self.activation(out)
        
        out = self.fc_2(out)
        #out = self.BN_2(out)
        out = self.Dropout(out)
        out = self.activation(out)
        
        out = self.fc(out)
        out = self.activation(out)
        out = out.view(-1)

        return out