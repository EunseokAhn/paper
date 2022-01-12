#%% module
import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time

print(torch.initial_seed())

torch.manual_seed(8602936841797904669)

#%%
def PrepareDataset(feature, BATCH_SIZE = 10, seq_len = 24, pred_len = 1, train_propotion = 0.7):
    time_len = feature.shape[0]
    
    max_val = feature.max().max()
    feature =  feature / max_val
    
    sequences, labels = [], []
    for i in range(time_len - seq_len - pred_len):
        sequences.append(feature.iloc[i:i+seq_len].values)
        labels.append(feature.iloc[i+seq_len:i+seq_len+pred_len].values)
    sequences, labels = np.asarray(sequences), np.asarray(labels)
    
    # shuffle and split the dataset to training and testing datasets
    sample_size = np.array(sequences).shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.shuffle(index)
    
    train_index = int(np.floor(sample_size * train_propotion))
    
    train_data, train_label = sequences[:train_index], labels[:train_index]
    test_data, test_label = sequences[train_index:], labels[train_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    return train_dataloader, test_dataloader, max_val


#%%
feature = pd.read_csv("/Users/an-eunseog/Desktop/bike_paper/data/Edge_num_mat2.csv")
feature_mat = torch.tensor(feature.to_numpy(dtype = "float32"))

train_dataloader, test_dataloader, max_val = PrepareDataset(feature.T)

#%% LSTM

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last = True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.output_last = output_last
        
    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        
        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs
    
    def initHidden(self, batch_size):
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State

#%%
inputs, labels = next(iter(train_dataloader))
[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size
#%%
model = LSTM(input_dim, hidden_dim, output_dim, output_last = True)

loss_MSE = torch.nn.MSELoss()
loss_L1 = torch.nn.L1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001)
model.train()
losses_train = []

losses_epoch_train = []
for epoch in range(50):
    trained_number = 0    
    losses_epoch_train = []   
    for data in train_dataloader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        model.zero_grad()
        outputs = model(inputs)
        loss_train = loss_MSE(outputs, torch.squeeze(labels))
        losses_train.append(loss_train.data)
        losses_epoch_train.append(loss_train.data)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        trained_number += 1
            
    avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
    losses_epoch_train.append(avg_losses_epoch_train)
    print("Epoch: {:.4f}".format(epoch), "MSE: {:.4f}".format(np.around(avg_losses_epoch_train, decimals=8)))


losses_mse = []
losses_l1 = [] 
losses_r2 = [] 
pred_lstm = []
 
for data in test_dataloader:
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = None
    outputs = model(inputs)
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    loss_mse = loss_MSE(outputs, torch.squeeze(labels))
    loss_l1 = loss_L1(outputs, torch.squeeze(labels))
    loss_r2 = 1 - (torch.mean((outputs-torch.squeeze(labels))**2)/torch.mean((torch.squeeze(labels)-torch.mean(torch.squeeze(labels)))**2))
    pred_lstm = np.append(pred_lstm, outputs.data.numpy())
    losses_mse = np.append(losses_mse, loss_mse.data.numpy())
    losses_l1 = np.append(losses_l1, loss_l1.data.numpy())
    losses_r2 = np.append(losses_r2, loss_r2.data.numpy())
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    losses_r2 = np.array(losses_r2)
    mean_rmse = np.sqrt(np.mean(losses_mse) * (max_val**2))
    mean_l1 = np.mean(losses_l1) * max_val
    mean_r2 = np.mean(losses_r2)
print('Tested: RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}'.format(mean_rmse, mean_l1, mean_r2))
    
# %%
# pred_lstm.shape
# lstm_pred = pred_lstm* max_val
# pred_df = pd.DataFrame(lstm_pred) 
# pred_df.to_csv('lstm_pred.csv', index=False)
