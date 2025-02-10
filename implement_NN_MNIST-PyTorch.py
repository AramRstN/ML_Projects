import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ipywidgets as widgets
from IPython.display import display
from matplotlib import pyplot as plt
from tqdm import trange

#dataset

train_data = datasets.MNIST(
    root = "data",
    traiin = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)


#DLs

batch_size = 64

train_dataloader = DataLoader(train_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#model

class NeuralNetwork (nn.modules):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward (self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train (model, optimzar, epochs = 10):
    losses, accuracies = [], []

    for _ in (pbar := trange(epochs)):
        r_loss, acc = 0, 0
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            predict = model(x)
            loss = loss_function(predict, y)
            optimizaer.step()
            optimizaer.zero_grad()
            r_loss += loss.item()
            predict = torch.argmax(predict, axis = 1)
            acc += sum(predict == y).item()
        
        acc /= len(train_dataloader.dataset)
        acc *= 100

        r_loss /= len(train_dataloader)
        losses.append(r_loss)
        accuracies.append(acc)
        pbar.set_description(f'Loss={r_loss:.3f} | Accuracy = {acc:.2f}%')
    return losses, accuracies

#init. the instance of the model
model = NeuralNetwork().to(device)

#loss_function
loss_function = nn.CrossEntropyLoss()

#optimizer
optimizaer = torch.optim.SGD(model.parameters(), lr = 1e-3)

#Train
epochs = 50

for _ in (pbar := trange(epochs)):

    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        optimizaer.zero_grad()
        predict =  model(x)
        loss = loss_function(predict, y)

        loss.backward()
        optimizaer.step()

        pbar.set_description(f'Loss = {loss.item():.3f}')


correct, total = 0, 0

with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pre = torch.argmax(logits, axis = 1)
        correct += sum(pre == y).item()
        total += pre.shape[0]

print(f'Accuracy:{100 * correct / total:.2f}')

