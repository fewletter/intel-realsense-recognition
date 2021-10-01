# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:37:35 2021

@author: fewle
"""

import pandas as pd 
import joblib
import os 
import numpy as np

from sklearn.preprocessing import LabelBinarizer

video_paths = os.listdir('C:/Users/fewle/Documents/dataset/npy/contour/')
print(f'The number of the mp4 file:{len(video_paths)}')

labels = np.array([int(x) for x in range(1,15)])
datavideo = pd.DataFrame()
c = 0

for video_path in video_paths:
    datavideo.loc[c,'video_path'] = f'C:/Users/fewle/Documents/dataset/npy/contour/{video_path}'
    video_floor = video_path.split('(',)[0]
    if video_floor == '1':
        datavideo.loc[c,'label'] = labels[0]
    elif video_floor == '2':
        datavideo.loc[c,'label'] = labels[1]
    elif video_floor == '3':
        datavideo.loc[c,'label'] = labels[2]
    elif video_floor == '4':
        datavideo.loc[c,'label'] = labels[3]
    elif video_floor == '5':
        datavideo.loc[c,'label'] = labels[4]
    elif video_floor == '6':
        datavideo.loc[c,'label'] = labels[5]
    elif video_floor == '7':
        datavideo.loc[c,'label'] = labels[6]
    elif video_floor == '8':
        datavideo.loc[c,'label'] = labels[7]
    elif video_floor == '9':
        datavideo.loc[c,'label'] = labels[8]
    elif video_floor == '10':
        datavideo.loc[c,'label'] = labels[9]
    elif video_floor == '11':
        datavideo.loc[c,'label'] = labels[10]
    elif video_floor == '12':
        datavideo.loc[c,'label'] = labels[11]
    elif video_floor == 'b1':
        datavideo.loc[c,'label'] = labels[12]
    elif video_floor == 'b2':
        datavideo.loc[c,'label'] = labels[13]
    c += 1
lb = LabelBinarizer()
labels = lb.fit_transform(labels) 

datavideo.to_csv('dataframe/datavideo.csv')

print('Saving the binarized labels as pickled file')
joblib.dump(lb, 'dataframe/lbelevator.pkl')
 
print(datavideo.head(5))

