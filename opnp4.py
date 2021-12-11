# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 23:27:41 2021

@author: fewle
"""

import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image


#opnp4_x = np.load('C:/Users/fewle/Documents/dataset/npy/test/depthc_test.npy')
opnp4_y = np.load('C:/Users/fewle/Documents/dataset/npy/test/depthc_test.npy')
#opnp4_z = np.load('C:/Users/fewle/Documents/dataset/npy/test/no8bit_test.npy')

#depth_image = opnp4_x[0]
frame_num = 0
no_deal = opnp4_y[frame_num].copy()
ebit_image = opnp4_y[frame_num]
#no_ebit_image = opnp4_z[0]

'''
depth_image = Image.fromarray(depth_image)
ebit_image = Image.fromarray(ebit_image)
no_ebit_image = Image.fromarray(no_ebit_image)
'''

w = []
for j in range(opnp4_y.shape[0]):
    w.append(np.where(opnp4_y[j] != 0)[1])
    #data_0 = plt.hist(w[0],range=(0,640),bins=640)
    data_all = plt.hist(w[j],range=(0,640),bins=640)
plt.show()
plt.imshow(ebit_image,cmap='gray')  
plt.show() 
 
ex_w = []
ex_w = np.where(ebit_image != 0)[1]
sum_ex = int(sum(ex_w)/len(ex_w))

ptr = np.where(ebit_image != 0)
ptr_2 = list(list(np.where(ptr[1] < 100))[0])

cor_y = []
cor_x = []
for k in range(len(ptr_2)):
    cor_y.append(ptr[0][int(ptr_2[k])])
    cor_x.append(ptr[1][int(ptr_2[k])])

np_cor_y = np.zeros((len(ptr_2)),np.uint8)
np_cor_y = np.asarray(cor_y)
np_cor_x = np.zeros((len(ptr_2)),np.uint8)
np_cor_x = np.asarray(cor_x)

ebit_image[cor_y,cor_x] = 0

a = ebit_image
z = Image.fromarray(a)
plt.subplot(1,2,1)
plt.imshow(z,cmap='gray')   
plt.subplot(1,2,2) 
plt.imshow(no_deal,cmap='gray')   
z = np.asarray(z)
plt.show()
