import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init

from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torchvision import datasets
from torchvision.transforms import ToTensor

from customAdam import NoBiasCorrectionAdam

##############################################
# Create CNN for MNIST
##############################################
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(5*5*64, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

##############################################
# Functions
##############################################
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)

def train(model, train_dataloader, criterion, optimizer, num_epochs=100):
    train_loss_list = []
    val_loss_list = []
    for _ in trange(num_epochs):
        # Train
        train_loss = 0
        val_loss = 0
        for _, (images, labels) in enumerate(train_dataloader):
            model.train()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Sum loss
            train_loss += loss.item()

        # Save loss
        train_loss_list.append(train_loss/len(train_dataloader))

        # Test
        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Sum loss
                val_loss += loss.item()

        # Save loss
        val_loss_list.append(val_loss/len(test_dataloader))

    return train_loss_list, val_loss_list

##############################################
# Main
##############################################
if __name__ == '__main__':
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Create model, loss function, and optimizer
    model1 = CNN()  
    model1.apply(weight_init)
    model2 = CNN()  
    model1.apply(weight_init)
    model3 = CNN()  
    model1.apply(weight_init)
    model4 = CNN()  
    model4.apply(weight_init)

    criterion = nn.CrossEntropyLoss()

    optimizer1 = Adam(model1.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)                     # Original
    optimizer2 = NoBiasCorrectionAdam(model2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)     # Original w/o bias correction
    optimizer3 = Adam(model3.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)                  # Oringinal w/ L2 regularization
    optimizer4 = AdamW(model4.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)                 # AdamW weight decay

    num_epochs = 100

    train_loss1, val_loss1 = train(model1, train_dataloader, criterion, optimizer1, num_epochs)
    train_loss2, val_loss2 = train(model2, train_dataloader, criterion, optimizer2, num_epochs)
    train_loss3, val_loss3 = train(model3, train_dataloader, criterion, optimizer3, num_epochs)
    train_loss4, val_loss4 = train(model4, train_dataloader, criterion, optimizer4, num_epochs)  

    # Plot training loss
    plt.plot(train_loss1, label='Adam')
    plt.plot(train_loss2, label='Adam w/o bias correction')
    plt.plot(train_loss3, label='Adam w/ L2 regularization')
    plt.plot(train_loss4, label='AdamW weight decay')
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('assets/training_loss.png')

    # Plot validation loss
    plt.clf()
    plt.plot(val_loss1, label='Adam')
    plt.plot(val_loss2, label='Adam w/o bias correction')
    plt.plot(val_loss3, label='Adam w/ L2 regularization')
    plt.plot(val_loss4, label='AdamW weight decay')
    plt.legend()
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('assets/validation_loss.png')


