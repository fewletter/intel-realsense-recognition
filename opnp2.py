# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 21:04:54 2021

@author: fewle
"""

import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
import random

#file = np.load('C:/Users/fewle/Documents/dataset/npy/l_point/point_3_4_1.npy')
#file = np.load('C:/Users/fewle/Documents/dataset/npy/l_binary/binary_3_4_1.npy')
file = np.load('C:/Users/fewle/Documents/dataset/npy/test/show.npy')
#y = np.load('E:/dataset/binary1280/6(10).npy')
'''
num_frames = file.shape[0]
part = random.randint(0, 10)
height_low = []
height_high = []
width_left = []
width_right = []

for i in range(16):
    b = file[i]
    #height,wide = b.shape
    #b = b.astype(float)
    
    z = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(z.copy(),cv2.COLOR_BGR2GRAY)
    
    #blur = cv2.GaussianBlur(z, (5,5), 10)        
    
    ret,thresh = cv2.threshold(b,10,255,0)#黑白影片所以閥值設10跟255
    contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 600*300 and cv2.contourArea(cnt) > 60*60:
            new_cnts = cnt
            x,y,w,h = cv2.boundingRect(new_cnts)
            cv2.rectangle(z,(x,y),(x+w,y+h),(0,255,0),5)
            height_high.append(y)
            height_low.append(y+h)
            width_left.append(x)
            width_right.append(x+w)

x_left = min(width_left)
x_right = max(width_right)
y_low = min(height_high)
y_high = max(height_low)

pad_h = 640-(y_high - y_low)
pad_w = 640-(x_right - x_left)
part1 = random.randint(0,pad_h)
part2 = random.randint(0,pad_w)

for j in range(16):
    video = file[j]
    b = Image.fromarray(video)
    
    x_left = width_left[j]
    x_right = width_right[j]
    y_low = height_high[j]
    y_high = height_low[j]
    
    #b = b.crop((x_left, y_low, x_right, y_high))

    b = np.asarray(b)
    #b = np.pad(b,((part1,pad_h-part1),(part2,pad_w-part2)),'constant',constant_values=((0,0),(0,0)))
    n, bins, patches = plt.hist( b, bins = 10, density=True, color='green')
    
    b = Image.fromarray(b)
    #b = b.resize((224,224))
    
    plt.subplot(4,4,j+1)
    plt.imshow(b,cmap='gray')

plt.show()
'''


b = file[17]
a = np.min(b)
n, bins, patches = plt.hist( b, bins = 10, density=True)
plt.xlabel("number")
plt.ylabel("number of number")
plt.title("depth pixel histogram")
plt.show()

