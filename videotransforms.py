# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:55:08 2021

@author: fewle
"""

import numpy as np
import numbers
import random
import torch
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as F
import cv2

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomPadResizecrop(object):
    def __init__(self):
        '''
        if data is contour,binary o_size = 640
        if use opnp2.py to open the npfile o_size = 1280
        '''
        self.o_size = 640
        self.size = 224
    
    def __call__(self,images):
        images = np.array(images)
        c,t,h,w = images.shape
        clips = []
        #紀錄手型座標極值
        height_high = []
        height_low = []
        width_left = []
        width_right= []
        if c == 1:
            images = np.transpose(images,[1,2,3,0])#clips[1,64,360,640] to clips[64,360,640,1]
            images = np.squeeze(images,axis=3)#clips[64,360,640] [t,h,w]
            '''
            run every frame,each frame size is (360,640),get bounding rectangle(x,y,x+w,y+h)
            '''
            for frame in range(t):
                ret,thresh = cv2.threshold(images[frame],10,255,0)#黑白影片所以閥值設10跟255
                contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cnts = sorted(contours, key=cv2.contourArea)
                for cnt in cnts:
                    if cv2.contourArea(cnt) < 600*300 and cv2.contourArea(cnt) > 60*60:
                        new_cnts = cnt
                        x,y,w,h = cv2.boundingRect(new_cnts)
                        height_high.append(y)
                        height_low.append(y+h)
                        width_left.append(x)
                        width_right.append(x+w) 
                
            #找出極值中的最大最小值
            x_left = min(width_left)
            x_right = max(width_right)
            y_low = min(height_high)
            y_high = max(height_low)
            
            #告訴圖片要pad多少
            pad_h = self.o_size-(y_high - y_low)
            pad_w = self.o_size-(x_right - x_left)
            part1 = random.randint(0,pad_h)
            part2 = random.randint(0,pad_w)
            
            #進行randompad
            #run every frame,get the randompad frame
            for clip in range(t):
                b = Image.fromarray(images[clip])
                b = b.crop((x_left, y_low, x_right, y_high))   
                pic = np.asarray(b)
                
                pic = np.pad(pic,((part1,pad_h-part1),(part2,pad_w-part2)),'constant')
                pic = Image.fromarray(pic)
                
                #進行resize
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
                
            clips = np.expand_dims(clips, axis=0)#clips sizes back to [1,64,360,640]
            
        '''
        if c == 3:
            a = np.pad(images,((0,0),(0,0),(int((w-h)/self.num)*part1,int((w-h)/self.num)*(self.num-part1)),(0,0)),'constant',constant_values=((0,0),(0,0),(160,160),(0,0)))
            images = np.transpose(a,[1,2,3,0])
            for frame in range(t):
                pic = Image.fromarray(images[frame])
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
            clips = np.asarray(clips)
        '''
        return clips
        
    def __repr__(self):
        return self.__class__.__name__

class testPadResizecrop(object):
    def __init__(self):
        self.size = 224
    
    def __call__(self,images):

        images = np.array(images)
        c,t,h,w = images.shape
        clips = []
        if c == 1:
            a = np.pad(images,((0,0),(0,0),(int((w-h)/2),int((w-h)/2)),(0,0)),'constant')
            images = np.transpose(a,[1,2,3,0])
            images = np.squeeze(images,axis=3)
            for frame in range(t):
                pic = Image.fromarray(images[frame])
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
            clips = np.expand_dims(clips, axis=0)
        if c == 3:
            a = np.pad(images,((0,0),(0,0),(int((w-h)/2),int((w-h)/2)),(0,0)),'constant',constant_values=((0,0),(0,0),(160,160),(0,0)))
            images = np.transpose(a,[1,2,3,0])
            for frame in range(t):
                pic = Image.fromarray(images[frame])
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
            clips = np.asarray(clips)
            clips = np.transpose(clips,[3,0,1,2])
        return clips
        
    def __repr__(self):
        return self.__class__.__name__ 

if __name__ == "__main__":
    inputs = torch.rand(1, 1, 36, 640, 640)
    train_transforms = RandomHorizontalFlip(inputs)
    outputs = transforms.Compose(train_transforms)