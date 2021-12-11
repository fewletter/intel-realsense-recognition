# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:40:35 2021

@author: fewle
"""

import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image
import os
import random
import videotransforms
from torchvision import transforms
import torch
import math

x = np.load('C:/Users/fewle/Documents/dataset/npy/l_ebit_depth/depth_1_b2_10.npy')
depth = np.load('C:/Users/fewle/Documents/dataset/npy/l_depth/depthraw_1_b2_10.npy')
last = x[-1]
last_r = np.repeat(np.expand_dims(last,axis=0),90-x.shape[0],axis=0)
new_video = np.append(x,last_r,axis=0)

num = new_video.shape[0]

for i in range(49):
    a = new_video[4+i]
    #h,w = a.shape
    #a = np.pad(a,((int((w-h)/10)*part1,int((w-h)/10)*(10-part1)),(0,0)),'constant')
    z = Image.fromarray(a)
    #z = z.resize((224,224),Image.BICUBIC)
    plt.subplot(7,7,i+1)
    plt.imshow(z,cmap='gray')
    z = np.asarray(z)
plt.show()
    
np.save('C:/Users/fewle/Documents/dataset/npy/test/new_video.npy',new_video)
'''
train_transforms = transforms.Compose([videotransforms.RandomPadResizecrop()])
x = np.expand_dims(x,axis=0)

t_images = train_transforms(x)
t_images = np.asarray(t_images)

t_squeeze = np.squeeze(t_images,axis=0)

for count in range(16):
    #a = t_images[count]
    a = t_squeeze[count]
    #h,w,_ = a.shape
    #a = np.pad(a,((int((w-h)/10)*part1,int((w-h)/10)*(10-part1)),(0,0)),'constant')
    z = Image.fromarray(a)
    plt.subplot(4,4,count+1)
    plt.imshow(z,cmap='gray')
    z = np.asarray(z)
plt.show()
'''