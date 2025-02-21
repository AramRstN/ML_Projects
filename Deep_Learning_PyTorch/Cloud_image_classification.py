import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics import Accuracy, Recall, Presicion

#### Image data

class Net (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 3, padding= 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Conv2d(32, 64, kernel_size= 3, padding= 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64*16*16, num_classes)

    def foward (self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

#Augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64,64))
])

test_transforms = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

#Multi-class classification
data_train = ImageFolder(
    "clouds_train",
    transform= train_transforms,
)

dataloader_train = DataLoader(
    data_train,
    shuffle= True,
    batch_size= 1
)

data_test = ImageFolder(
    "cloud_test",
    transform= test_transforms
)
dataloader_test = DataLoader(
    data_test,
    shuffle= True,
    batch_size= 1
)
image, label = next(iter(dataloader_train))
image = image.squeeze().permute(1,2,0)


net = Net(num_classes = 7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters, lr= 0.001)
epochs = 10

#training loop
def train_model(optimizer, net, num_epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        # Iterate over training batches
        for images, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader_train)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

##evaluation
        
### Recall:
#recall_per_class = Recall(task= 'multiclass', num_classes= 7, average = None)
#recall_micro = Recall(task= 'multiclass', num_classes= 7, average = "micro") -> imbalanced data
#recall_Macro = Recall(task= 'multiclass', num_classes= 7, average = "macro") -> care about performance on small classes
#recall_weighted = Recall(task= 'multiclass', num_classes= 7, average = "weighted") -> consider error in large classes as more important
            
metric_precision =Presicion(
    task= "multiclass", num_classes= 7, average= "macro"
)
metrics_recall = Recall(
    task= "multiclass", num_classes= 7, average= "macro"
)
net.eval()
with torch.no_grad():
    for image, labels in dataloader_test:
        outputs = net(image)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metrics_recall(preds, labels)

presicion = metric_precision.compute()
recall = metrics_recall.compte()

print(f"Presicion: {presicion}")
print(f"Recall: {recall}")

### in case of all classes metrics:
## Get precision per class
# precision_per_class = {
#     k: presicion[v].item()
#     for k, v 
#     in data_test.class_to_idx.items()
# }
# print(precision_per_class)



