#first input = written alphabet as image
#second input = language that it comes from as a vector [0,...,1,0,...,0]

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
    def __init__(self):
        super().__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size= 2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        self.alphabet_layer = nn.Sequential(
            nn.Linear(30, 8),
            nn.ELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128+8, 964)
        )
    
    def forward(self, x_image, x_alphabet):
        x_image = self. image_layer(x_image)
        x_alphabet = self.alphabet_layer(x_alphabet)
        x = torch.cat((x_image,x_alphabet), dim=1)
        return self.classifier(x)
    

dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=3,
)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.01)

##Train
num_epochs = 10
for epoch in range(num_epochs):
    for img, alpha, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model(img, alpha)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()