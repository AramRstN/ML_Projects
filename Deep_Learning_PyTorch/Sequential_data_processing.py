### Electerisity Consumption Presiction (RNN - sequence to vector)
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.nn import nn


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

# RNN
class net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN (
                input = 1, #Electrisity consumption
                hidden_size= 32,
                num_layers= 2,
                batch_first= True
        )
        self.fc= nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        out,_ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
#in time series, for avoiding "look-ahead bias" -> split data by time