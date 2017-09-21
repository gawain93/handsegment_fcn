# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:38:30 2017

@author: dhri-dz
"""

import cv2
import os
import numpy as np

class input_pre(object):
    
    def __init__(self, input_path, num_channels):
        self.input_path = input_path
        self.num_channels = num_channels
        
        self.img_list = sorted(os.listdir(self.input_path + '/rgb'))
        self.mean_rgb = np.zeros(self.num_channels)
        for file in self.img_list:
            img = cv2.imread(self.input_path + '/rgb/' + file)
            avr = np.mean(img, axis = 0)
            avr = np.mean(avr, axis = 0)
            self.mean_rgb = self.mean_rgb + avr            
        self.mean_rgb = self.mean_rgb / len(self.img_list)
        
        self.std_rgb = np.zeros(self.num_channels)
        mean_mat = np.ones([256,256,3])
        mean_mat[...,0] = mean_mat[...,0] * self.mean_rgb[0]
        mean_mat[...,1] = mean_mat[...,1] * self.mean_rgb[1]
        mean_mat[...,2] = mean_mat[...,2] * self.mean_rgb[2]
        for f in self.img_list:
            img = cv2.imread(self.input_path + '/rgb/' + f)
            img = img - mean_mat
            squ = np.square(img)
            squ = squ.sum(axis = 0)
            squ = squ.sum(axis = 0)
            self.std_rgb = self.std_rgb + squ
        self.std_rgb = self.std_rgb /( img.shape[0]*img.shape[1]*len(self.img_list))           
        self.std_rgb = np.sqrt(self.std_rgb)
        
   
        ## TODO 
    
    def global_normalization_rgb(self, batch):
        normalized_batch = np.zeros(batch.shape)
        for i in range(self.num_channels):
            channel_data = batch[...,i]
            normalized_batch[...,i] = (channel_data - self.mean_rgb[i]) / (self.std_rgb[i] + 1e-9)
            
        return normalized_batch
    
    @staticmethod    
    def normalization_rgb(batch):
        batch_min = batch.min()
        batch_max = batch.max()
        batch_normalized = 2. * (batch - batch_min) / (batch_max - batch_min) - 1.
        
        return batch_normalized
        
        