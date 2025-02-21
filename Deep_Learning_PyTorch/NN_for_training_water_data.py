import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from torch.nn.init import init

#### Tabular data

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        
        #He initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity = 'sigmoid')


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)

        # Pass x through the second set of layers
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        
        x = nn.functional.sigmoid(self.fc3(x))
        return x
    
# Create an instance of the WaterDataset
dataset_train = WaterDataset('water_train.csv')
dataset_test = WaterDataset('water_test.csv')

# Create a DataLoaders
dataloader_train = DataLoader(
    dataset_train,
    batch_size= 2,
    shuffle=True,
)

dataloader_test = DataLoader(
    dataset_test,
    batch_size= 2,
    shuffle=True,
)

# Get a batch of features and labels
features, labels = next(iter(dataloader_train))
print(features, labels)

net = Net()
# SGD, RMSprop, Adam
optimizer = optim.SGD(net.parameters(), lr= 0.01)
epochs = 10

#training loop
def train_model(optimizer, net, num_epochs):
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        for feature, label in dataloader_train:
            optimizer.zero_grad()
            output = net(feature)
            loss = criterion(output, label.view(-1,1))
            loss.backward()
            optimizer.step()

##evaluation
            
acc = Accuracy(task = "binary")

net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        outputs = net(features)
        preds = (outputs >= 0.5).float()
        acc(preds, labels.view(-1, 1))

test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")