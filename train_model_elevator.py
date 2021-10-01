# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:55:12 2021

@author: fewle
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d
from pytorch_c3d import C3D

from elevator_dataset import elevator_dataset as Dataset


batch_size = 1

train_dataset = Dataset('dataframe/datavideo.csv', 'train', 'binary' , transform=None )
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset('dataframe/datavideo.csv', 'validation', 'binary' , transform=None)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)    

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
datasets = {'train': train_dataset, 'val': val_dataset}

print('dataset ready!')
'''
# setup the model
if mode == 'flow':
    i3d = InceptionI3d(400, in_channels=2)
    i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
else:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
'''

i3d = InceptionI3d(14, in_channels=1)
#i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
i3d.replace_logits(14)
#i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
i3d.cuda()
i3d = nn.DataParallel(i3d)

print('i3d model ready!')

init_lr = 0.1
lr = init_lr
optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

num_epochs = 150
correct = 0
val_correct = 0
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []
# train it
for epoch in range(num_epochs):
        print ('Step {}/{}'.format(epoch+1, num_epochs))
        print ('There will be {} files used in training , {} files used in validation'.format(len(train_dataloader)*batch_size,len(val_dataloader)*batch_size))
        print ('-' * 10)
        
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        val_tot_loss = 0.0
        
        optimizer.zero_grad()
        for i,data in enumerate(train_dataloader):
            #print('第'+str(epoch+1)+'個training epoch')
            #print('第'+str(i)+'個mini epoch')
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            inputs = inputs.float()
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            labels = labels.float()
            
            i3d.train(True)
            outputs = i3d(inputs)

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(outputs, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(outputs, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()
            
            loss = 0.5*loc_loss + 0.5*cls_loss
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_sched.step()
            
            sigmoid = nn.Sigmoid()
            sigmoid_o = sigmoid(outputs.cpu())
            _,predict = torch.max(sigmoid_o,axis=1)
            predict = predict.detach().numpy()#size = (1,7)
            predict_number = np.argmax(np.bincount(np.squeeze(predict,axis=0)))#squeeze size to(7)
            target = labels.cpu()
            target = target.detach().numpy()#size = (1,14,7)
            target_number = np.argmax(np.bincount(np.argmax(np.squeeze(target,axis=0),axis=0)))
            if predict_number == target_number:
                correct += 1

            if i == int(0.5*len(train_dataloader)):
               print('mini_batch = {} training accuarcy: {:.4f} BCE training Loss: {:.4f}'.format(i*batch_size,correct*2/len(train_dataloader),tot_loss*2/len(train_dataloader)))
            #if data is l_binary i=183
            elif i == len(train_dataloader)-1:
               print('mini_batch = {} training accuarcy: {:.4f} BCE training Loss: {:.4f}'.format(i*batch_size,correct/len(train_dataloader),tot_loss/len(train_dataloader)))
               train_accuracy.append(correct/len(train_dataloader))
               train_loss.append(tot_loss/len(train_dataloader))
               # save model
               #torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
               tot_loss = tot_loc_loss = tot_cls_loss =0 
                
        for j,data in enumerate(val_dataloader):
            #print('第'+str(epoch+1)+'個validation epoch')
            #print('第'+str(j)+'個mini epoch')
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            inputs = inputs.float()
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            labels = labels.float()
               
            i3d.train(False)
            val_outputs = i3d(inputs)
            
             # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(val_outputs, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(outputs, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()

            loss = 0.5*loc_loss + 0.5*cls_loss
            val_tot_loss += loss.item()
            
            sigmoid = nn.Sigmoid()
            sigmoid_o = sigmoid(val_outputs.cpu())
            _,predict = torch.max(sigmoid_o,axis=1)
            predict = predict.detach().numpy()
            predict_number = np.argmax(np.bincount(np.squeeze(predict,axis=0)))
            val_target = labels.cpu()
            target = val_target.detach().numpy()
            target_number = np.argmax(np.bincount(np.argmax(np.squeeze(target,axis=0),axis=0)))
            if predict_number == target_number:
                val_correct += 1
            
            if j == len(val_dataloader)-1:
               print('test accuarcy: {:.4f} test Loss: {:.4f} '.format(val_correct/len(val_dataloader),val_tot_loss/len(val_dataloader)))
               val_accuracy.append(val_correct/len(val_dataloader))
               val_loss.append(val_tot_loss/len(val_dataloader))
               tot_loss = tot_loc_loss = tot_cls_loss = val_tot_loss = 0. 
        
        correct = 0
        val_correct = 0
        torch.save(i3d.module.state_dict(), 'elevator/epoch150_t8_binary_randompad_model.pth')

np.save('elevator/picture/train_accuracy.npy',np.asarray(train_accuracy))
np.save('elevator/picture/val_accuracy.npy',np.asarray(val_accuracy))
np.save('elevator/picture/train_loss.npy',np.asarray(train_loss))
np.save('elevator/picture/val_loss.npy',np.asarray(val_loss))

'''
if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
'''