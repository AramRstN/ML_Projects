import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchmetrics import Accuracy

class OmniglotDataset (Dataset):
    def __init__(self, trasform, samples):
        #samples are tuples of 3, created from data file path
        self.transform = trasform
        self.samples = samples

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        image_path, alphabet, label = self.samples[idx]
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        return img, alphabet, label
    
class Net(nn.Module):
    def __init__(self, num_alpha, num_char):
        super().__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size= 2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        self.classifier_alpha = nn.Linear(128, 30)
        self.classifier_char = nn.Linear(128, 964)
    
    def forward(self, x):
        x_image = self. image_layer(x_image)
        output_alphabet = self.classifier_alpha(x_image)
        output_char = self.classifier_char(x_image)
        return output_alphabet, output_char
# Datasets
    
dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
      	transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

dataloader_train = DataLoader(
    dataset_train,
    shuffle= True,
    batch_size= 1
) 
dataset_test = []
dataloader_test = []
# training
    
net = Net()
criterion = nn.MESLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
num_epochs = 10

for epochs in range(num_epochs):
    for images, labels_alpha, labels_char in dataloader_train:
        optimizer.zero_grad()
        output_alphabet, output_char = net(images)
        loss_alpha = criterion(output_alphabet, labels_alpha)
        loss_char = criterion(output_char, labels_char)
        loss = loss_alpha + loss_char
        loss.backward()
        optimizer.step()


#evaluation:
    
acc_alpha = Accuracy(task= "multiclass", num_classes= 30)
acc_char = Accuracy(task="multiclass", num_classes= 964)

net.eval()
with torch.no_grad():
    for images, labels_alpha, labels_char in dataloader_test:
        out_alpha, out_char = net(images)
        _, pred_alpha = torch.max(out_alpha, 1)
        _, pred_char = torch.max(out_char, 1)
        acc_alpha(pred_alpha, labels_alpha)
        acc_char(pred_char, labels_char)
    
    print(f"Alphabet:{acc_alpha.compute()}")
    print(f"Character:{acc_char.compute()}")