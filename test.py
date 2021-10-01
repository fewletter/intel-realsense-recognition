# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:50:02 2021

@author: fewle
"""

import torch
import numpy as np
import joblib
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
import train_model
import albumentations as A
from torchvision.transforms import transforms   
from torch.utils.data import Dataset, DataLoader
from PIL import Image

print('Loading model and label binarizer...')
lb = joblib.load('dataframe/lb.pkl')

model = train_model.CustomCNN().cuda()
print('Model loading...')

model.load_state_dict(torch.load('model/saved_model.pth'))
print('Pretrained model loading...')

aug = A.Compose([
    A.Resize(224, 224),
    ])

cap = cv2.VideoCapture('C:/Users/fewle/Videos/Captures/chess.mp4')

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        model.eval()
        with torch.no_grad():
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = aug(image=np.array(pil_image))['image']
            pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
            pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
            pil_image = pil_image.unsqueeze(0)
            outputs = model(pil_image)
            _, preds = torch.max(outputs.data, 1)
            
        cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.imshow('image', frame)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else: 
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()