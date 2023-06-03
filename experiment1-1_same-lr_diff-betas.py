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
from sklearn.metrics import classification_report
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
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
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Sum accuracy
                predicted = torch.max(outputs.data, 1)[1]
                total_test += len(labels)
                correct_test += (predicted == labels).float().sum()
                
                all_labels.extend(labels.tolist())
                all_preds.extend(predicted.tolist())
                
                # Sum loss
                val_loss += loss.item()
                
        # Calculate the confusion matrix
        cm = confusion_matrix(all_labels, all_preds, binary=False)
        
        # Calculate the accuracy, precision, recall, F1 score
        report = classification_report(all_labels, all_preds, output_dict=True)

        # Save accuracy
        val_accuracy = 100 * correct_test / float(total_test)
        val_accuracy_list.append(val_accuracy.cpu())
        
        # Save loss
        val_loss_list.append(val_loss/len(test_dataloader))

    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, report, cm

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
    
    lr_list = [1e-3, 1e-4, 1e-5]
    betas_list = [(0.9, 0.999), (0.99, 0.999)]
    
    for lr in tqdm(lr_list):
        
        train_loss1s, val_loss1s, train_acc1s, val_acc1s = [], [], [], []
        train_loss2s, val_loss2s, train_acc2s, val_acc2s = [], [], [], []
        
        for betas in tqdm(betas_list):
            model1 = CNN().to(device)
            model1.apply(weight_init)
            model2 = CNN().to(device)  
            model2.apply(weight_init)

            optimizer1 = Adam(model1.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=0)                     # Original
            optimizer2 = NoBiasCorrectionAdam(model2.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=0)     # Original w/o bias correction

            num_epochs = 100

            train_loss1, val_loss1, train_acc1, val_acc1, classification_report1, confusion_matrix1 = train(model1, train_dataloader, criterion, optimizer1, num_epochs)
            train_loss2, val_loss2, train_acc2, val_acc2, classification_report2, confusion_matrix2 = train(model2, train_dataloader, criterion, optimizer2, num_epochs)

            train_loss1s.append(train_loss1)
            val_loss1s.append(val_loss1)
            train_acc1s.append(train_acc1)
            val_acc1s.append(val_acc1)

            train_loss2s.append(train_loss2)
            val_loss2s.append(val_loss2)
            train_acc2s.append(train_acc2)
            val_acc2s.append(val_acc2)

            # Save classification_report1
            classification_report1_DF = pd.DataFrame(classification_report1).transpose()
            classification_report1_DF.to_csv(f'runs/experiment1-1/adam_with_bias_correction{lr:.0e}-{betas}-classification_report.csv', index=True)
            
            # Save classification_report2
            classification_report2_DF = pd.DataFrame(classification_report2).transpose()
            classification_report2_DF.to_csv(f'runs/experiment1-1/adam_without_bias_correction{lr:.0e}-{betas}-classification_report.csv', index=True)

            # Save training loss results
            trainLossDF = pd.DataFrame({
                'Adam': train_loss1,
                'Adam w/o bias correction': train_loss2,
            })
            trainLossDF.to_csv(f'runs/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-training_loss.csv', index=False)

            # Save training accuracy results
            trainAccDF = pd.DataFrame({
                'Adam': train_acc1,
                'Adam w/o bias correction': train_acc2,
            })
            trainAccDF.to_csv(f'runs/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-training_accuracy.csv', index=False)

            # Save validation loss results
            validationLossDF = pd.DataFrame({
                'Adam': val_loss1,
                'Adam w/o bias correction': val_loss2,
            })
            validationLossDF.to_csv(f'runs/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-validation_loss.csv', index=False)

            # Save validation accuracy results
            validationAccDF = pd.DataFrame({
                'Adam': val_acc1,
                'Adam w/o bias correction': val_acc2,
            })
            validationAccDF.to_csv(f'runs/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-validation_accuracy.csv', index=False)

            # Plot confusion_matrix1 figure 
            plt.clf()
            fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix1)
            plt.title("Confusion Matrix")
            plt.savefig(f"assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-confusion_matrix.png")

            # Plot confusion_matrix2 figure 
            plt.clf()
            fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix2)
            plt.title("Confusion Matrix")
            plt.savefig(f"assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-{betas}-confusion_matrix.png")


        ############  Plot Adam with & without Bias Correction Training Loss & Accuracy  ############
        # Plot Adam with & without bias correction training loss with different betas
        plt.clf()
        for i, train_loss1 in enumerate(train_loss1s):
            plt.plot(train_loss1, label=f'Adam_betas={betas_list[i]}')
        for i, train_loss2 in enumerate(train_loss2s):
            plt.plot(train_loss2, label=f'Adam_w/o_BiasCorr_betas={betas_list[i]}')
        plt.legend()
        plt.title(f'Adam w/ & w/o Bias Correction Training Loss with LR={lr:.0e} & Different Betas')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Loss')
        plt.savefig(f'assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-training_loss.png')

        # Plot Adam with & without bias correction training accuracy with different betas
        plt.clf()
        for i, train_acc1 in enumerate(train_acc1s):
            plt.plot(train_acc1, label=f'Adam_betas={betas_list[i]}')
        for i, train_acc2 in enumerate(train_acc2s):
            plt.plot(train_acc2, label=f'Adam_w/o_BiasCorr_betas={betas_list[i]}')
        plt.legend()
        plt.title(f'Adam w/ & w/o Bias Correction Training Accuracy with LR={lr:.0e} & Different Betas')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Accuracy')
        plt.savefig(f'assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-training_accuracy.png')
        
        ############  Plot Adam with & without Bias Correction Validation Loss & Accuracy  ############
        # Plot Adam with & without bias correction validation loss with different betas
        plt.clf()
        for i, val_loss1 in enumerate(val_loss1s):
            plt.plot(val_loss1, label=f'Adam_betas={betas_list[i]}')
        for i, val_loss2 in enumerate(val_loss2s):
            plt.plot(val_loss2, label=f'Adam_w/o_BiasCorr_betas={betas_list[i]}')
        plt.legend()
        plt.title(f'Adam w/ & w/o Bias Correction Validation Loss with LR={lr:.0e} and Different Betas')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Loss')
        plt.savefig(f'assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-validation_loss.png')
        
        # Plot Adam with & without bias correction validation accuracy with different betas
        plt.clf()
        for i, val_acc1 in enumerate(val_acc1s):
            plt.plot(val_acc1, label=f'Adam_betas={betas_list[i]}')
        for i, val_acc2 in enumerate(val_acc2s):
            plt.plot(val_acc2, label=f'Adam_w/o_BiasCorr_betas={betas_list[i]}')
        plt.legend()
        plt.title(f'Adam w/ & w/o Bias Correction Validation Accuracy with LR={lr:.0e} and Different Betas')
        plt.xlabel('Epoch')
        plt.xticks(np.arange(0, num_epochs+1, 10))
        plt.ylabel('Accuracy')
        plt.savefig(f'assets/experiment1-1/adam_with_and_without_bias_correction-{lr:.0e}-validation_accuracy.png')