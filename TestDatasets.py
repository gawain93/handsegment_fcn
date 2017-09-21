# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:16:19 2017

@author: dhri-dz
"""

import os
import cv2
import numpy as np
from random import shuffle
from utils import utils

input_path = '/home/dhri-dz/savedata'
os.chdir('/home/dhri-dz/savedata')

'''
im_small = im[0:200, 200:400, :]
cv2.imshow("image",im_small)
cv2.waitKey(0)


value_max = im.max()
value_min = im.min()
value_range = value_max - value_min
        
scale = (im - value_min) / value_range
normalized_roi = scale 
'''

rgb_list = sorted(os.listdir(os.getcwd() + "/rgb"))
label_list = sorted(os.listdir(os.getcwd() + "/mask"))
depth_list = sorted(os.listdir(os.getcwd() + "/depthmap"))
slist = [i for i in range(len(rgb_list))]
shuffle(slist)

batch_size = 20
num_step = ceil (len(slist) / batch_size) 
object_size = (128, 128)
batch_counter = 0
uti = utils(object_size, input_path)

for i in range(len(slist)):
            
    #while batch_counter < batch_size:
                
          rgb = cv2.imread(input_path + '/rgb/' + rgb_list[i] , 1)
          #depth = cv2.imread(self.input_path + '/depthmap/' self.list[i] , 1)  for further use
          label = cv2.imread(input_path + '/mask/'+ label_list[i] , 1)
                
          rgb_roi = uti.cut_ROI(rgb, i)
          label_roi = uti.generate_label(label,i)
          
          cv2.imwrite("./rgbcut/" + rgb_list[i], rgb_roi)
          cv2.imwrite("./rgbcut/" + label_list[i], label_roi)
          #batch_counter += 1

output_path = '/home/dhri-dz/testfolder'
os.chdir(output_path)
import h5py

f = h5py.File('10.h5', 'r')

img_batches = f["imgs_for_training"][:]        
labels_temp = f["labels_for_training"][:] 
print img_batches.shape

test_img = img_batches[10,:,:,:]
labels_temp = labels_temp[:,:,:,0]
label = labels_temp[10,:,:]
cv2.imshow('image', label.astype(uint8))
cv2.waitKey(0)