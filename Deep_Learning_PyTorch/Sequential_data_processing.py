### Electerisity Consumption Presiction (RNN - sequence to vector)
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.nn import nn
from torchmetrics import MeanSquaredError
import torch.optim as optim

sequence_length = 24*4
data_train = []

def create_sequence(df, seq_lenght):
    xs, ys = [],[]

    for i in range(len(df) - seq_lenght):
        x = df.iloc[i:(i-seq_lenght),1]
        y = df.iloc[i+seq_lenght, 1]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

X_train, Y_train = create_sequence(data_train, sequence_length)

dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(Y_train).float()
)

#make the dataloaders ready!
dataloader_train = []
dataloader_test = []


# RNN with LSTM (it can be also GRU)
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM (
                input_size = 1, #Electrisity consumption
                hidden_size= 32,
                num_layers= 2,
                batch_first= True
        )
        self.fc= nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        c0 = torch.zeros(2, x.size(0), 32)
        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])
        return out
    
net = Net()
criterion = nn.MESLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

def train_model (model, num_epochs):
    for epochs in range(num_epochs):
        for seqs, labels in dataloader_train:
            seqs = seqs.view(32, 96, 1)
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.forward()
            optimizer.step()

#evaluation
mse = torchmetrics.MeanSquaredError()     
def eval(model):
    model.eval()
    with torch.no_grad():
        for seqs, labels in dataloader_test:
            seqs = seqs.view(32, 96, 1)
            outputs = model(seqs).squeeze()
            mse(outputs, labels)
    print(f"Test MSE: {mse.compute()}")

#in time series, for avoiding "look-ahead bias" -> split data by time