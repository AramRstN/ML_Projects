import os

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tqdm import tqdm
import torchvision.transforms as transforms



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Local response normalization
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Local response normalization
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False), # Dropout layer for regularization
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False), # Dropout layer for regularization
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes), # Output layer for classification
        )
        self.init_bias() # Initializes the biases of the model layers



    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)



    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)
        return self.classifier(x)
    
class AlexNetFashionMNIST(nn.Module):
    def __init__(self, num_classes=10):  
        super(AlexNetFashionMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            # No further pooling to maintain feature map size at 7x7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(256 * 7 * 7, 4096),  # Adjusted for 7x7 input
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(4096, 1024),  # Reduced for computational efficiency
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),  # Output for 10 classes in Fashion-MNIST
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def create_model():
    return AlexNetFashionMNIST(num_classes=NUM_CLASSES).to(device) 
   
def create_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

def create_criterion():
    return nn.CrossEntropyLoss()

def get_smaller_dataset(dataset, fraction=0.1):
    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, int(dataset_size * fraction), replace=False)
    return Subset(dataset, indices)

def augment_dataset(dataset):
    """Expand the dataset by adding horizontally and vertically flipped versions of the images."""
    horizontal_flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])
    vertical_flip_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()])

    # Augment the dataset by flipping
    horizontal_flipped_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=horizontal_flip_transform)
    vertical_flipped_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=vertical_flip_transform)
    # Combine the original, horizontally flipped, and vertically flipped datasets
    augmented_dataset = ConcatDataset([dataset, horizontal_flipped_dataset, vertical_flipped_dataset])
    return augmented_dataset

def load_data(augment=False, batch_size=64, fraction=0.1):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load full Fashion-MNIST dataset and subsample them based on the given fraction
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    small_trainset = get_smaller_dataset(trainset, fraction)

    # If augment=True, augment the dataset with horizontal and vertical flips
    if augment:
        small_trainset = augment_dataset(small_trainset)

    trainloader = DataLoader(small_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # calculate the loss
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # update the parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
        print(f"Finished Epoch {epoch+1}")

def evaluate_model(model, testloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc="Evaluating")
    with torch.no_grad():  # Disable gradient computation for testing
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # Predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            progress_bar.set_postfix(loss=test_loss / (i + 1), accuracy=100 * correct / total)

    # Calculate average test loss, accuracy, F1 score, and recall 
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {test_loss / len(testloader):.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, Recall Score: {recall:.4f}")
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}")

def run_experiment(augment=False, epochs=10, fraction=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model, criterion, and optimizer
    model = create_model()
    criterion = create_criterion()
    optimizer = create_optimizer(model)

    # Load data with the specified fraction
    trainloader, testloader = load_data(augment=augment, fraction=fraction)

    # Train model
    print(f"Training with {'augmentation' if augment else 'no augmentation'} on a {fraction * 100}% subset of Fashion-MNIST")
    train_model(model, trainloader, criterion, optimizer, device, epochs=epochs)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, testloader, criterion, device)

# without augmentation
print("Running training on Fashion-MNIST without augmentation:")
run_experiment(augment=False, epochs=4, fraction=0.1)  # Using 10% of the dataset

# with augmentation
print("\n\nRunning training on Fashion-MNIST with augmentation:")
run_experiment(augment=True, epochs=4, fraction=0.1)  # Using 10% of the dataset
