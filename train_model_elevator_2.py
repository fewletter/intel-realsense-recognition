# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:35:24 2021

@author: fewle
"""

import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import videotransforms
from elevator_dataset import elevator_dataset as Dataset

import pytorch_c3d
from pytorch_c3d import C3D

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 50  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

# setup dataset
train_transforms = transforms.Compose([videotransforms.RandomCrop(112),
                                       videotransforms.RandomHorizontalFlip(),
    ])
test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])

train_dataset = Dataset('dataframe/datavideo.csv', 'train', 'rgb' , train_transforms )
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

val_dataset = Dataset('dataframe/datavideo.csv', 'validation', 'rgb' , test_transforms )
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)    

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
datasets = {'train': train_dataset, 'val': val_dataset}

model = C3D(num_classes=14,pretrained=False)
train_params = [{'params': pytorch_c3d.get_1x_lr_params(model), 'lr': lr},
                {'params': pytorch_c3d.get_10x_lr_params(model), 'lr': lr * 10}]

criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
optimizer = optim.SGD(train_params, lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

model.to(device)
criterion.to(device)

trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

for epoch in range(nEpochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]


            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            
            running_loss = 0.0
            running_corrects = 0.0


            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")


'''
if __name__ == "__main__":
    train_model()
'''