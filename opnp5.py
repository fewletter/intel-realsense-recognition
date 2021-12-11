# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:21:12 2021

@author: fewle
"""

import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image

video = np.load('C:/Users/fewle/Documents/dataset/npy/test/new_video.npy')
num = video.shape[0]

for i in range(90):
    a = video[i]
    z = Image.fromarray(a)
    plt.subplot(10,9,i+1)
    plt.imshow(z,cmap='gray')
    z = np.asarray(z)
plt.show()