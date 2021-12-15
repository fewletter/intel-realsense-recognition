# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:53:38 2021

@author: fewle
"""

import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import csv
import h5py
import random
import os
import os.path
from sklearn.model_selection import train_test_split
from PIL import Image
import videotransforms
from torchvision import transforms
from  matplotlib import pyplot as plt

transform = transforms.Compose([videotransforms.RandomPadResizecrop()])

def load_frame(csv_dir_lines,mode):
    clips = []
    video = np.load(csv_dir_lines)
    resize_video = []
    pad_video = []
    fps = video.shape[0]
    if mode == 'rgb':
        '''
        for j in range(fps):
            image = video[j]
            h,w,_ = image.shape
            pad = np.pad(image,((w-h,0),(0,0),(0,0)),'constant',constant_values=((160,0),(0,0),(0,0)))
            image = Image.fromarray(pad)
            padimage = image.resize((224,224))
            resize_image = np.array(padimage)
            resize_video.append(resize_image)
        resize_video = np.asarray(resize_video)
        '''
        if fps < 4:
            repeat_time = int(4/fps)
            con_fps = 4%fps
            new_frame1 = np.tile(video,(repeat_time,1,1,1)) 
            new_frame2 = video[:con_fps]
            new_video = np.concatenate([new_frame1,new_frame2])
            new_fps = new_video.shape[0]
            for i in range(new_fps):
                frame = new_video[i]
                clips.append(frame)
            return clips
        elif fps >= 4:
            for i in range(4):
                frame = video[i]
                clips.append(frame)
            return clips
        
    elif mode == 'binary':
        
        for j in range(fps):
            image = video[j]
            image = Image.fromarray(image)
            image = image.resize((640,360))
            image = np.asarray(image)
            #image = np.transpose(image,[1,0])
            resize_video.append(image)
        resize_video = np.asarray(resize_video)
        
        if fps < 10:
            repeat_time = int(10/fps)
            con_fps = 10%fps
            new_frame1 = np.tile(resize_video,(repeat_time,1,1)) 
            new_frame2 = resize_video[:con_fps]
            new_video = np.concatenate([new_frame1,new_frame2])
            new_fps = new_video.shape[0]
            for i in range(new_fps):
                frame = new_video[i]
                clips.append(frame)
            clips = np.asarray(clips)
            return clips
        elif fps >= 10:
            for i in range(10):
                frame = resize_video[i]
                clips.append(frame)
            clips = np.asarray(clips)
            return clips
        


def load_csv_file(csv_file):
    video_dir = []
    label = []
    with open(csv_file) as csvfile:
        lines = csv.reader(csvfile)
        
        for line in lines:
            
            video_dir.append(line[1])
            label.append(line[2])
            
    return video_dir,label

def create_dataset(csv_file,mode):
    dataset = []
    video_dir,labels = load_csv_file(csv_file)
    '''
    for i in range(1,len(video_dir)):
        clips = load_frame(video_dir[i],mode)
        
        #cross entropy , 64 frames label_size=[1,7] 
        
        label = np.zeros((1,7),np.float32)
        number = float(labels[i])
        label[0,:] = number 
        dataset.append((clips,label))
    '''
    new_dataset = []
    for times in range(3):
        for i in range(1,len(video_dir)):
            clips = load_frame(video_dir[i],mode)
            trans_clips = np.expand_dims(clips, axis=0)
            trans_clips = transform(trans_clips)

            trans_label = np.zeros((7),np.float32)
            number = float(labels[i])
            trans_label[:] = number-1 
            new_dataset.append((trans_clips,trans_label))
        times += 1
    '''    
    l = []
    l = dataset + new_dataset
    '''
    return new_dataset

def split_dataset(csv_file,data_type,mode):
    train_dataset,val_dataset = train_test_split(create_dataset(csv_file,mode),test_size=0.1,random_state=13)
    if data_type == 'train':
        return train_dataset
    elif data_type == 'validation':
        return val_dataset
    
one = create_dataset('dataframe/datavideo.csv', 'binary')
'''
videoline,line = load_csv_file('dataframe/datavideo.csv')
clips = load_frame(videoline[6], 'rgb')

train_dataset = split_dataset('dataframe/datavideo.csv','train','binary')
val_dataset = split_dataset('dataframe/datavideo.csv','validation','binary')

plt.figure()
for i in range(4):
        a = clips[i]  
        image = Image.fromarray(a)
        image = image.resize((224,224))
        plt.subplot(2,2, i+1)
        plt.imshow(image)
plt.show()
'''
    
class elevator_dataset(data_utl.Dataset):
    def __init__(self, csvfile , data_type , mode , transform ):
        self.data = split_dataset(csvfile, data_type, mode)
        self.mode = mode
        self.transform = transform
    
    def __getitem__(self,index):     
        clips,label = self.data[index]
        
        #clips = np.asarray(clips)
        '''
        if self.mode == 'rgb':
            clips = np.transpose(clips,[3,0,1,2])
        else :
            clips = np.expand_dims(clips,axis=0)
            
        if self.transform == None: 
            clips = clips
        else:
            clips = self.transform(clips)
        '''
        
        clips = torch.from_numpy(clips)
        label = np.array(label)
        label = torch.from_numpy(label)
        return clips,label
    
    def __len__(self):
        return(len(self.data))

   
train_data = elevator_dataset('dataframe/datavideo.csv','train','binary',transform=None)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

val_data = elevator_dataset('dataframe/datavideo.csv','validation','binary',transform=None)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)

for i,(data,label) in enumerate(train_dataloader):
    if i == 1000:
        data_1 = data
    if i == 1001:
        data_2 = data
    print(data,label)

data_1 = np.squeeze(data_1,axis=0)
data_1 = np.squeeze(data_1,axis=0)
data_1 = data_1.detach().numpy()

data_2 = np.squeeze(data_2,axis=0)
data_2 = np.squeeze(data_2,axis=0)
data_2 = data_2.detach().numpy()

image1 = Image.fromarray(data_1[0])
image2 = Image.fromarray(data_2[0])
plt.subplot(2,1,1)
plt.imshow(image1,cmap='gray')
plt.subplot(2,1,2)
plt.imshow(image2,cmap='gray')
plt.show()
