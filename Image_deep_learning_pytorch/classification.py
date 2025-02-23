from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional
from torchvision.datasets import ImageFolder
import torch.nn as nn
from PIL import Image
import PIL

train_dir = '/data/train'
train_dataset = ImageFolder(root=train_dir, transform=transforms.ToTensor())

classes = train_dataset.classes
print(classes)
print(train_dataset.class_to_idx)

#num_channels
image = PIL.Image.open('dog.png')
num_channels = functional.get_image_num_classes(image)
print(num_channels)

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*112*112, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        x = self.sigmoid(x)
        return x
    
class MultiClassCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*112*112, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        x = self.softmax(x)
        return x
    
