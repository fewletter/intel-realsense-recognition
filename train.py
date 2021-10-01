# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 18:18:54 2021

@author: fewle
"""

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import albumentations
import torch.optim as optim
import os
import train_model
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

matplotlib.style.use('ggplot')

args = argparse.ArgumentParser()
args.add_argument('-e','--epochs',type=int,default=75,help='number of epochs to train our network for')
args = vars(args.parse_args())

lr = 1e-3
batch_size = 32
device = 'cuda:0'
print(f"Computation device:{device}\n")

df = pd.read_csv('dataframe/data.csv')
X = df.image_path.values
Y = df.target.values
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.1,random_state=42)

print(f"Training instances: {len(xtrain)}")
print(f"Validation instances: {len(xtest)}")

class ImageDataset(Dataset):
    def __init__(self,images,labels=None,tfms=None):
        self.X = images
        self.Y= labels
        
        if tfms == 0:
            self.aug = albumentations.Compose([albumentations.Resize(224,224,always_apply=True)])
        else :
            self.aug = albumentations.Compose([albumentations.Resize(224,224,always_apply=True),
                                               albumentations.HorizontalFlip(p=0.5),
                                               albumentations.ShiftScaleRotate(
                                                   shift_limit=0.3,
                                                   scale_limit=0.3,
                                                   rotate_limit=15,
                                                   p=0.5
                                                   ),])
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = image.convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.Y[i]
        return (torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long))

train_data = ImageDataset(xtrain,ytrain)
test_data = ImageDataset(xtest,ytest)

trainloader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
testloader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

model = train_model.CustomCNN().to(device)
total_parameters = sum(p.numel()for p in model.parameters())
print(f'{total_parameters:,}total_parameters')
total_trainable_parameters = sum(p.numel()for p in model.parameters()if p.requires_grad)
print(f'{total_trainable_parameters:},total_trainable_parameters')

optimizer = optim.Adam(model.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=True
    )

def fit(model, train_dataloader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_dataloader), total=int(len(train_data)/train_dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
    
    return train_loss, train_accuracy

def validate(model, test_dataloader):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=int(len(test_data)/test_dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
        
        return val_loss, val_accuracy

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss, train_epoch_accuracy = fit(model, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(model, testloader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    scheduler.step(val_epoch_loss)
end = time.time()
print(f"{(end-start)/60:.3f} minutes")

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model/accuracy.png')
plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model/loss.png')
plt.show()
    
# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), 'model/saved_model.pth')
 
print('TRAINING COMPLETE')
