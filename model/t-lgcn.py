#%% module
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from torch_geometric_temporal.nn.recurrent import LRGCN
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from sklearn.metrics import mean_squared_error

print(torch.initial_seed())

torch.manual_seed(8602936841797904669)

#%% data
feature = pd.read_csv("/Users/an-eunseog/Desktop/bike_paper/data/Edge_num_mat2.csv")
edge = pd.read_csv("/Users/an-eunseog/Desktop/bike_paper/data/edge_ind2.csv")

edges = np.array(edge).T
edge_weights = np.ones(edges.shape[1])


stacked_target = np.array(feature).T
stacked_target.shape

max_val = stacked_target.max()
min_val = stacked_target.min()
scaled_stacked_target = (stacked_target - min_val)/(max_val-min_val)

slw = max_val-min_val 

lag = 24

features = [
    scaled_stacked_target[i : i + lag, :].T
    for i in range(scaled_stacked_target.shape[0] - lag)
    ]

targets = [
    scaled_stacked_target[i + lag, :].T
    for i in range(scaled_stacked_target.shape[0] - lag)
    ]

dataset = StaticGraphTemporalSignal(
    edges, edge_weights, features, targets
    )

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)


#%% TGCN
hidden_unit = 100

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, hidden_unit)
        self.linear = torch.nn.Linear(hidden_unit, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
model = RecurrentGCN(node_features = lag)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(700):
    cost = 0
    mae1 = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        mae1 = mae1 + torch.mean(torch.abs(y_hat-snapshot.y))
    cost = cost / (time+1)
    mae1 = mae1 / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad() 
    print("Epoch: {:.4f}".format(epoch), "MSE: {:.4f}".format(cost), "MAE: {:.4f}".format(mae1))

print(torch.sqrt(cost*(slw**2)))
print(mae1*slw)

pred = []  
    
model.eval()
rmse = 0
mae = 0
r2 = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    pred = np.append(pred, y_hat.data.numpy())
    rmse = rmse + torch.sqrt(torch.mean((y_hat-snapshot.y)**2))
    mae = mae + torch.mean(torch.abs(y_hat-snapshot.y))
    r2 = r2 + 1 - (torch.mean((y_hat-snapshot.y)**2)/torch.mean((snapshot.y-torch.mean(snapshot.y))**2))
rmse = rmse * slw / (time+1)
mae  = mae * slw / (time+1)
r2  = r2 / (time+1)
rmse = rmse.item()
mae = mae.item()
print("RMSE: {:.4f}".format(rmse), "MAE: {:.4f}".format(mae), "r2: {:.4f}".format(r2))


#pred = pred*slw
#pred_df = pd.DataFrame(pred)
#pred_df.to_csv('pred_TLGCN.csv', index=False)

"""
hyper parameter
train
lr = 0.001, epoch = 100, lag = 24
  hidden unit = 4 : rmse = 8.5285, mae = 8.1658
  hidden unit = 8 : rmse = 1.3128, mae = 0.8667
  hidden unit = 16 : rmse = 1.3704, mae = 0.8954 
  hidden unit = 32 : rmse = 1.2844, mae = 0.8177 
  hidden unit = 64 : rmse = 1.2783, mae = 0.8046
  hidden unit = 100 : rmse = 1.2686, mae = 0.7864
  hidden unit = 128 : rmse = 1.2802, mae = 0.8035
  
lr = 0.001, epoch = 200, lag = 24
  hidden unit = 4 : rmse = 1.5758, mae = 1.1583
  hidden unit = 100 : rmse = 1.2629, mae = 0.7786 
lr = 0.001, epoch = 300, lag = 24
  hidden unit = 100 : rmse = 1.2608, mae = 0.7754
lr = 0.001, epoch = 400, lag = 24
  hidden unit = 100 : rmse = 1.2597, mae = 0.7734
lr = 0.001, epoch = 500, lag = 24
  hidden unit = 100 : rmse = 1.2592, mae = 0.7751
lr = 0.001, epoch = 600, lag = 24
  hidden unit = 100 : rmse = 1.2589, mae = 0.7726 
lr = 0.001, epoch = 700, lag = 24
  hidden unit = 100 : rmse = 1.2586, mae = 0.7734 ***
lr = 0.001, epoch = 800, lag = 24
  hidden unit = 100 : rmse = 1.2583, mae = 0.7773   
lr = 0.001, epoch = 900, lag = 24
  hidden unit = 100 : rmse = 1.2580, mae = 0.7701  
lr = 0.001, epoch = 1000, lag = 24
  hidden unit = 100 : rmse = 1.2578, mae = 0.7682   
"""  

"""
test

lr = 0.001, epoch = 700, lag = 24
  hidden unit = 100 : rmse = 1.2953, mae = 0.8740 ***
"""
 



