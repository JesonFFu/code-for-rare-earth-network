import numpy as np
import collections
import os
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from numpy.random import Generator, RandomState
from scipy.fft import fft, ifft
from scipy.integrate import solve_ivp
import scipy.io
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import pandas as pd

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ',device)

seed = 0
torch.manual_seed(seed)  # 
np.random.seed(seed)      # 
random.seed(seed)         # 
import sys
print(sys.version)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def control_data_precision(x,j=5):
    x = np.array(x)
    x = np.round(x, decimals=j)
    return x
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



transform = transforms.Compose([
    transforms.ToTensor(),
])


saved_data = torch.load('E:\\data2\\1018.pt')

rare_train_data = saved_data['rare_train_data']
rare_train_labels = saved_data['rare_train_labels']

rare_test_data = saved_data['rare_test_data']
rare_test_labels = saved_data['rare_test_labels']


rare_train_tensor = TensorDataset(
    torch.stack(rare_train_data),  
    torch.tensor(rare_train_labels)  
)


rare_train_loader = DataLoader(rare_train_tensor, batch_size=32, shuffle=True)



rare_test_tensor = TensorDataset(
    torch.stack(rare_test_data),  
    torch.tensor(rare_test_labels)  
)


rare_test_loader = DataLoader(rare_test_tensor, batch_size=32, shuffle=False)


x = torch.randn(28, 28)


x_flat = x.flatten()  

class RidgeLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg  
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, model):
        mse_loss = self.mse(y_pred, y_true)
        
        
        l2_reg = torch.tensor(0., device=y_pred.device)
        for param in model.parameters():
            if param.dim() > 1:  
                l2_reg += torch.norm(param, p=2) ** 2
        
        total_loss = mse_loss + self.lambda_reg * l2_reg
        return total_loss

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        
        self.output = nn.Linear(input_size, 10)

    def forward(self, x):
        x = x.view(-1, 784) 

        x = self.output(x)


        return x
    

input_size = int(784)  
model = MLP(input_size)
model = model.to(device)  
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0005)


num_epochs = 70
print(f"trainable parameters: {count_parameters(model)}")  



best_acc = 0.0
best_model_path = 'best_mlp_model.pth' 


train_losses = []
acc_1=[]
acc_train=[]
test_losses=[]
def train(epoch):
    epoch_loss = 0.0
    correct = 0
    total = 0
    model.train()
    running_loss = 0.0
    test_loss=0.0
    for batch_idx, (data, target) in enumerate(rare_train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target_onehot = torch.zeros(target.size(0), 10)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)   
        target_onehot = target_onehot.to(device)     
        target = target.to(device)
        data=torch.tensor(data,dtype=torch.float32)
        
        output = model(data)
        loss = criterion(output, target_onehot)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)


        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(rare_train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')    
    epoch_loss = running_loss / len(rare_train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch}: Loss = {epoch_loss:.4f}')

    print(f'Epoch {epoch} completed. Avg Loss: {epoch_loss/len(rare_train_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')
    re_1=test_accuracy()
    acc_1.append(re_1)
    acc_train.append(100 * correct / total)


def test_accuracy():
    global best_acc 
    global best_model_path 

    model.eval()
    correct = 0
    total = 0
    running_loss2=0.0
    with torch.no_grad():
        for data, target in rare_test_loader:
            data = data.to(device)
            target = target.to(device) 
            target_onehot = torch.zeros(target.size(0), 10)
            target_onehot = target_onehot.to(device)  
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)   
            data=torch.tensor(data,dtype=torch.float32)
                       
            output = model(data)
            loss2 = criterion(output, target_onehot)

            running_loss2 += loss2.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    current_acc = 100 * correct / total
    print(f'test accuracy: {current_acc:.2f}%')   
    epoch_loss2 = running_loss2/ len(rare_test_loader)
    test_losses.append(epoch_loss2)

   
    if current_acc > best_acc:
        print(f"current test accuracy {current_acc:.2f}% better than history {best_acc:.2f}%,save model to {best_model_path}")
        torch.save(model.state_dict(), best_model_path)
        best_acc = current_acc
    return current_acc


for epoch in range(1, num_epochs + 1):
    train(epoch)
    

print(f"\nfinish,best test accuracy: {best_acc:.2f}%")


print(f"load best model: {best_model_path}")
model.load_state_dict(torch.load(best_model_path))
model.to(device) 



plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss') 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.savefig('loss_curves.png')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(acc_train, label='Training Accuracy')
plt.plot(acc_1, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Curve')
plt.legend()
plt.savefig('accuracy_curves.png')
plt.show()




def plot_confusion_matrix_with_best_model():
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in rare_test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Best Model)')
    plt.savefig('confusion_matrix_best_model.png')
    plt.show()

print("\nprint best confusion matrix:")
plot_confusion_matrix_with_best_model()

b=1