from matplotlib import pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

torch.manual_seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_epoch(model, dataloader, loss_func, optimizer, device):
    model.train()
    train_loss = 0.
    train_acc = 0. 

    for image, lebel in dataloader:
        image, label = image.to(device), label.to(device)
        logits = model(image)
        loss = loss_func(logits, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_acc += (logits.argmax(dim=1) == label).sum().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)

    return train_loss, train_acc

def validation_epoch(model, dataloader, loss_func, device):
    model.eval()
    val_loss = 0.
    val_acc = 0.

    with torch.inference_mode():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            logist = model(image)
            loss = loss_func(logist, label)

            val_loss += loss.item()
            val_acc += (logist.argmax(dim=1) == label).sim().item()

        val_loss /= len(dataloader)
        val_acc /= len(dataloader.dataset)

    return val_loss, val_acc

def train_model (model, train_dl, val_dl, optimizer, epochs, device = device):
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[],'val_acc':[]}
    loss_func = CrossEntropyLoss()

    for _ in (pbar := trange(epochs)):

        train_loss, train_acc = train_epoch(model, train_dl, loss_func, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc = validation_epoch(model, val_dl, loss_func, device)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(val_acc)

        pbar.set_description(f'Training Accuracy {100* train_acc:.2f}% | Validation Accuracy {100* val_acc:.2f}%')

    return history


#Visualization:
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'], label='val')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()

#dataset--- CIFAR10
    
norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)
batch_size = 128
    
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
        


#### Image to Vector Embedding 
#Patch Embedding
class PatchEmbedding(nn.Module):

    def __init__ (self, image_size, p_size, channels, h_size):
        super().__init__()

        self.patches = (image_size // p_size) ** 2
        self.projection = nn.Conv2d(channels, h_size, kernel_size= p_size, stride= p_size)

    def forward(self, x):
        
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
# combine embeddings
class Embeddings (nn.Module):
    def __init__ (self, image_size, p_size, channels, h_size, h_dropout_prob):
        super(.).__init__()

        self.patch_embedding = PatchEmbedding(image_size, p_size, channels, h_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, h_size))
        self.pos_e = nn.parameter(torch.randn(1, self.patch_embedding.patches + 1, h_size))
        self.dropout = nn.Dropout(h_dropout_prob)

        def forward(self, x):
            x = self.patch_embedding(x)
            batch_size, _, _ = x.size()
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_e
            x = self.dropout(x)

            return x
        
# ViT Classification Model

class ViT (nn.Module):

    def __init__(self, image_size, p_size, channels, h_size, heads, layers, mlp_dim, classes, droupout_prob = 0.1):
        self.embedding = Embeddings(image_size, p_size, channels, h_size, droupout_prob)

        #Transformet Encoder
        encoder = nn.TransformerEncoderLayer(
            d_model= h_size,
            nhead= heads,
            dim_feedforward= mlp_dim,
            dropout= droupout_prob,
            activation= 'gelu',
            batch_first= True
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers= layers)

        # Classification with MLP
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(h_size),
            nn.Linear(h_size, classes)
        )

    def forward (self, x):
        x = self.embedding(x)
        x = self.encoder(x)

        cls_token = x[:, 0]

        logits = self.mlp_head(cls_token)

        return logits
    