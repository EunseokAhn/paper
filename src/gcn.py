#%% module
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

print(torch.initial_seed())

torch.manual_seed(8602936841797904669)

# %%
feature = pd.read_csv("/Users/an-eunseog/Desktop/bike_paper/data/Edge_num_mat2.csv")
edge = pd.read_csv("/Users/an-eunseog/Desktop/bike_paper/data/edge_ind2.csv")
feature.shape

edge_index = torch.tensor(np.array(edge).T, dtype=torch.long)

feature1 = np.array(feature)
max_val = feature1[:984,1439].max()
feature2 = feature1/max_val

x1 = torch.tensor(feature2[:,966:990], dtype=torch.float)
y1 = torch.tensor(feature2[:,990], dtype=torch.float)

train_data = Data(x=x1, y=y1, edge_index=edge_index)


#%%
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(train_data.num_node_features, 64)
        self.conv2 = GCNConv(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
    
# %%

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(train_data)
    loss = torch.mean((out-train_data.y)**2)
    loss.backward()
    optimizer.step()
    
rmse = 0
mae = 0
gcn_pred = []  

for i in range(425):
    x2 = torch.tensor(feature2[:,990+i:1014+i], dtype=torch.float)
    y2 = torch.tensor(feature2[:,1014+i], dtype=torch.float)
    test_data = Data(x=x2, y=y2,edge_index=edge_index)
    model.eval()
    pred = model(test_data)
    gcn_pred = np.append(gcn_pred, pred.data.numpy())
    rmse = rmse + torch.sqrt(torch.mean((pred-test_data.y)**2) * (max_val**2))
    mae = mae + torch.mean(torch.abs(pred-test_data.y)) * max_val
mean_rmse = rmse / 425
mean_mae = mae / 425
print('RMSE: {:.4f}'.format(mean_rmse), 'MAE: {:.4f}'.format(mean_mae))   
        

# %%
# gcn_pred.shape
# gcn_pred = gcn_pred* max_val
# pred_df = pd.DataFrame(gcn_pred) 
# pred_df.to_csv('gcn_pred.csv', index=False)
