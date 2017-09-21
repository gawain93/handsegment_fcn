# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:38:58 2017

@author: dhri-dz
"""

import os
import cv2
from utils import utils
from keras.models import load_model
from keras.utils import np_utils
from input_pre import input_pre
from net_new import fcn32
import numpy as np
import pylab as plt

data_path = "/home/dhri-dz/train data"
model_path = '/home/dhri-dz/train data/my_model_870.h5'
test_size = 10
img_height = 256
img_width = 256
img_channels = 3
no_class = 2
ip = input_pre(data_path, img_channels)

os.chdir(data_path)
test_list = sorted(os.listdir(data_path + '/rgb'))[3150:3150+test_size]

test_batch = np.ndarray([test_size, img_height, img_width, img_channels])
test_label = np.ndarray([test_size, img_height, img_width, no_class])

for i in range(test_size):
        
    img2 = cv2.imread("./rgb/" + test_list[i],1)
    test_batch[i,...] = img2
    video_name = test_list[i].split('.')[0].split('_')[0]
    frame_num = test_list[i].split('.')[0].split('_')[2]
        
    label2 = cv2.imread('./label/' + video_name + '_label_' + frame_num + '.jpg', 1)
    label2 = np.around(label2 / 255.)
    label2 = label2[...,1]            # the three channels of the label matrix are the same
    test_label[i,:,:,0] = label2
    test_label[i,:,:,1] = 1 - label2
    
test_batch = ip.global_normalization_rgb(test_batch)  
model = fcn32(num_class = 2)
model.load_weights(model_path)
#model = load_model(model_path)
predicted1 = []
predicted1 = model.predict(test_batch)

''' below is used to compare the training results'''
test_rgb_mask = cv2.imread("/home/dhri-dz/train data/rgb/"+test_list[0])
test_rgb = test_rgb_mask
cv2.imwrite('/home/dhri-dz/Desktop/training result/rgb.jpg', test_rgb)
test_mask= test_label[0,:,:,0]
predict_mask = np.zeros(test_mask.shape[0:2])
p0=predicted1[0,:,:,0]
p1=predicted1[0,:,:,1]
for h in  range(test_rgb.shape[0]):
    for w in range(test_rgb.shape[1]):
        if p0[h,w] > 0.1: #p1[h,w]:
            predict_mask[h,w]=1
            test_rgb_mask[h,w,:] = 0
        else:
            predict_mask[h,w]=0
plt.imshow(predict_mask)
cv2.imshow('image',test_rgb)
cv2.imwrite('/home/dhri-dz/Desktop/training result/rgbwithmask.jpg', test_rgb_mask)
cv2.imwrite('/home/dhri-dz/Desktop/training result/groundtruth.jpg', test_mask*255)
cv2.imwrite('/home/dhri-dz/Desktop/training result/predictmask.jpg', predict_mask*255)