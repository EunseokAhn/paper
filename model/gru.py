#%%
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
# %%
class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    
    
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
         
       
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)
            

        out = outs[-1].squeeze()
        
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out
    
#%%    
inputs, labels = next(iter(train_dataloader))
[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size
# %%
model = GRUModel(input_dim, hidden_dim, 1,output_dim)

loss_MSE = torch.nn.MSELoss()
loss_L1 = torch.nn.L1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001)
model.train()
losses_train = []

losses_epoch_train = []
for epoch in range(20):
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
pred_gru = []
 
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
    pred_gru = np.append(pred_gru, outputs.data.numpy())
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
# pred_gru.shape
# gru_pred = pred_gru* max_val

# pred_df = pd.DataFrame(gru_pred) 
# pred_df.to_csv('gru_pred.csv', index=False)