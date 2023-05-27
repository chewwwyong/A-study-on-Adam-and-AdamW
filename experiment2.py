import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init

from tqdm import trange,tqdm
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
    train_accuracy_list = []
    val_accuracy_list = []
    for _ in trange(num_epochs):
        # Train
        train_loss = 0
        val_loss = 0
        correct_train = 0
        total_train = 0
        for _, (images, labels) in enumerate(train_dataloader):
            model.train()
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Sum accuracy
            predicted = torch.max(outputs.data, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()
            
            # Sum loss
            train_loss += loss.item()

        # Save accuracy
        train_accuracy = 100 * correct_train / float(total_train)
        train_accuracy_list.append(train_accuracy.cpu())
        
        # Save loss
        train_loss_list.append(train_loss/len(train_dataloader))

        # Test
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)

                # Sum accuracy
                predicted = torch.max(outputs.data, 1)[1]
                total_test += len(labels)
                correct_test += (predicted == labels).float().sum()
                
                # Sum loss
                val_loss += loss.item()

        # Save accuracy
        val_accuracy = 100 * correct_test / float(total_test)
        val_accuracy_list.append(val_accuracy.cpu())
        
        # Save loss
        val_loss_list.append(val_loss/len(test_dataloader))

    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list

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

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # Create model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    train_loss1s, val_loss1s, train_acc1s, val_acc1s = [], [], [], []
    train_loss2s, val_loss2s, train_acc2s, val_acc2s = [], [], [], []
    weight_decays = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    learning_rate = 1e-3
    for weight_decay in tqdm(weight_decays):

        model1 = CNN().to(device)
        model1.apply(weight_init)
        model2 = CNN().to(device) 
        model2.apply(weight_init)

        optimizer1 = Adam(model1.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        optimizer2 = AdamW(model2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

        num_epochs = 100

        train_loss1, val_loss1, train_acc1, val_acc1 = train(model1, train_dataloader, criterion, optimizer1, num_epochs)
        train_loss2, val_loss2, train_acc2, val_acc2 = train(model2, train_dataloader, criterion, optimizer2, num_epochs)
        
        train_loss1s.append(train_loss1)
        val_loss1s.append(val_loss1)
        train_acc1s.append(train_acc1)
        val_acc1s.append(val_acc1)
        
        train_loss2s.append(train_loss2)
        val_loss2s.append(val_loss2)
        train_acc2s.append(train_acc2)
        val_acc2s.append(val_acc2)
        
        # Save training loss results
        trainLossDF = pd.DataFrame({
            'Adam': train_loss1,
            'AdamW': train_loss2,
        })
        trainLossDF.to_csv(f'runs/experiment2/weight_decay={weight_decay:.0e}-training_loss.csv', index=False)
        
        # Save training accuracy results
        trainAccDF = pd.DataFrame({
            'Adam': train_acc1,
            'AdamW': train_acc2,
        })
        trainAccDF.to_csv(f'runs/experiment2/weight_decay={weight_decay:.0e}-training_accuracy.csv', index=False)
        
        # Save validation loss results
        validationLossDF = pd.DataFrame({
            'Adam': val_loss1,
            'AdamW': val_loss2
        })
        validationLossDF.to_csv(f'runs/experiment2/weight_decay={weight_decay:.0e}-validation_loss.csv', index=False)
        
        # Save validation accuracy results
        validationAccDF = pd.DataFrame({
            'Adam': val_acc1,
            'AdamW': val_acc2,
        })
        validationAccDF.to_csv(f'runs/experiment2/weight_decay={weight_decay:.0e}-validation_accuracy.csv', index=False)
    
     
    ############  Plot Adam and AdamW Training Loss & Accuracy  ############
    # Plot Adam training loss with different weight decay
    plt.clf()
    for i, train_loss1 in enumerate(train_loss1s):
        plt.plot(train_loss1, label=f'Adam_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'Adam Training Loss with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Loss')
    #plt.yticks(np.arange(0, 0.5, 0.1))
    plt.savefig(f'assets/experiment2/adam_diff_weight_decay-training_loss.png')
    
    # Plot AdamW training loss with different weight decay
    plt.clf()
    for i, train_loss2 in enumerate(train_loss2s):
        plt.plot(train_loss2, label=f'AdamW_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'AdamW Training Loss with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Loss')
    #plt.yticks(np.arange(0, 0.5, 0.1))
    plt.savefig(f'assets/experiment2/adamW_diff_weight_decay-training_loss.png')
    
    # Plot Adam training accuracy with different weight decay
    plt.clf()
    for i, train_acc1 in enumerate(train_acc1s):
        plt.plot(train_acc1, label=f'Adam_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'Adam Training Accuracy with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Accuracy')
    #plt.yticks(np.arange(98, 100, 0.25))
    plt.savefig(f'assets/experiment2/adam_diff_weight_decay-training_accuracy.png')
    
    # Plot AdamW training accuracy with different weight decay
    plt.clf()
    for i, train_acc2 in enumerate(train_acc2s):
        plt.plot(train_acc2, label=f'AdamW_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'AdamW Training Accuracy with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Accuracy')
    #plt.yticks(np.arange(98, 100, 0.25))
    plt.savefig(f'assets/experiment2/adamW_diff_weight_decay-training_accuracy.png')
    
    
    ############  Plot Adam and AdamW Validation Loss & Accuracy  ############
    # Plot Adam validation loss with different weight decay
    plt.clf()
    for i, val_loss1 in enumerate(val_loss1s):
        plt.plot(val_loss1, label=f'Adam_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'Adam Validation Loss with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Loss')
    #plt.yticks(np.arange(0, 0.5, 0.1))
    plt.savefig(f'assets/experiment2/adam_diff_weight_decay-validation_loss.png')
    
    # Plot AdamW validation loss with different weight decay
    plt.clf()
    for i, val_loss2 in enumerate(val_loss2s):
        plt.plot(val_loss2, label=f'AdamW_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'AdamW Validation Loss with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Loss')
    #plt.yticks(np.arange(0, 0.5, 0.1))
    plt.savefig(f'assets/experiment2/adamW_diff_weight_decay-validation_loss.png')
    
    # Plot Adam validation accuracy with different weight decay
    plt.clf()
    for i, val_acc1 in enumerate(val_acc1s):
        plt.plot(val_acc1, label=f'Adam_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'Adam Validation Accuracy with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Accuracy')
    #plt.yticks(np.arange(98, 100, 0.25))
    plt.savefig(f'assets/experiment2/adam_diff_weight_decay-validation_accuracy.png')
    
    # Plot AdamW validation accuracy with different weight decay
    plt.clf()
    for i, val_acc2 in enumerate(val_acc2s):
        plt.plot(val_acc2, label=f'AdamW_Weight_Decay={weight_decays[i]}')
    plt.legend()
    plt.title(f'AdamW Validation Accuracy with LR={learning_rate} and Different Weight Decay')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, num_epochs+1, 10))
    plt.ylabel('Accuracy')
    #plt.yticks(np.arange(98, 100, 0.25))
    plt.savefig(f'assets/experiment2/adamW_diff_weight_decay-validation_accuracy.png')
    
    
    
    ############  Plot Mixing Adam and AdamW Validation Loss & Accuracy  ############
    # Plot Mixing Adam and AdamW training loss with different weight decay
    plt.clf()
    for i, train_loss1 in enumerate(train_loss1s):
        plt.plot(train_loss1, label='Adam' if i == 0 else "", color='red')
   
    for i, train_loss2 in enumerate(train_loss2s):
        plt.plot(train_loss2, label='AdamW' if i == 0 else "", color='blue')
        plt.legend()
        plt.title(f'Adam and AdamW Training Loss with LR={learning_rate} and Different Weight Decay')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Loss')
        #plt.yticks(np.arange(0, 0.5, 0.1))
        plt.savefig(f'assets/experiment2/adam_adamW_diff_weight_decay-training_loss.png')
        
    # Plot Mixing Adam and AdamW training accuracy with different weight decay
    plt.clf()
    for i, train_acc1 in enumerate(train_acc1s):
        plt.plot(train_acc1, label='Adam' if i == 0 else "", color='red')
    for i, train_acc2 in enumerate(train_acc2s):
        plt.plot(train_acc2, label='AdamW' if i == 0 else "", color='blue')
        plt.legend()
        plt.title(f'Adam and AdamW Training Accuracy with LR={learning_rate} and Different Weight Decay')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Accuracy')
        #plt.yticks(np.arange(98, 100, 0.25))
        plt.savefig(f'assets/experiment2/adam_adamW_diff_weight_decay-training_accuracy.png')

    # Plot Mixing Adam and AdamW validation loss with different weight decay
    plt.clf()
    for i, val_loss1 in enumerate(val_loss1s):
        plt.plot(val_loss1, label='Adam' if i == 0 else "", color='red')
    for i, val_loss2 in enumerate(val_loss2s):
        plt.plot(val_loss2, label='AdamW' if i == 0 else "", color='blue')
        plt.legend()
        plt.title(f'Adam and AdamW Validation Loss with LR={learning_rate} and Different Weight Decay')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Loss')
        #plt.yticks(np.arange(0, 0.5, 0.1))
        plt.savefig(f'assets/experiment2/adam_adamW_diff_weight_decay-validation_loss.png')
        
   # Plot Mixing Adam and AdamW validation accuracy with different weight decay
    plt.clf()
    for i, val_acc1 in enumerate(val_acc1s):
        plt.plot(val_acc1, label='Adam' if i == 0 else "", color='red')
    for i, val_acc2 in enumerate(val_acc2s):
        plt.plot(val_acc2, label='AdamW' if i == 0 else "", color='blue')
        plt.legend()
        plt.title(f'Adam and AdamW Validation Accuracy with LR={learning_rate} and Different Weight Decay')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Accuracy')
        #plt.yticks(np.arange(97, 99, 0.25))
        plt.savefig(f'assets/experiment2/adam_adamW_diff_weight_decay-validation_accuracy.png')


