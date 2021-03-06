# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:12:37 2017

@author:Jiawen Xu
"""
'''
The class used for pre-process the datasets for hand segmentation using deep learning(FCN)
input: 
     input_path: the path to the data  generated by the dataset_creation , cpp project(default 480*640)
     output_path: the datapath of the output
     object_size: the size of the ROI (tuple)
     
The functions are only for a single frame
'''

import cv2
import os
import yaml
import numpy as np

class utils(object):
    
    def __init__(self, object_size, input_path):
        
        self.input_path = input_path
        self.object_size = object_size
        os.chdir(self.input_path)
        skip_lines = 2
        with open("coordinates_and_mask.yml", 'r') as self.f:
            for i in  range(skip_lines):
                _ = self.f.readline()
            self.utils_data = yaml.load(self.f)
        
        self.ori_height = 480
        self.ori_width =  640
        self.hand_middle_x = self.utils_data["Middle point of hand x"]
        self.hand_middle_y = self.utils_data["Middle point of hand y"]
        assert len(self.hand_middle_x) == len(self.hand_middle_y)
    
    def cut_ROI(self, img, img_index):
        
        x = self.hand_middle_x[img_index]
        y = self.hand_middle_y[img_index]
        #img_rgb = cv2.imread(self.input_path + str(self.image_index), 1)           #data_path needed to be changed
        img_rgb_padded = cv2.copyMakeBorder(img,self.object_size[0],self.object_size[0],self.object_size[1],self.object_size[1],cv2.BORDER_CONSTANT, value = 0) # if three channels?
        ROI = img_rgb_padded[y : y+2*self.object_size[1], x : x+2*self.object_size[0], :]
        
        return ROI
        
    def generate_label(self, img_label, label_index):
        
        x = self.hand_middle_x[label_index]
        y = self.hand_middle_y[label_index]
        #img_label = cv2.imread(self.input_path + str(self.image_index - 1 ), 1)  # -1 attention!! 0 based index
        img_label_padded = cv2.copyMakeBorder(img_label,self.object_size[0],self.object_size[0],self.object_size[1],self.object_size[1],cv2.BORDER_CONSTANT, value = 0) # if three channels?
        label_roi = img_label_padded[y : y+2*self.object_size[1], x : x+2*self.object_size[0], :]
        
        return label_roi
    
    @staticmethod
    def normalization(input_roi):    
        
        value_max = input_roi.max()
        value_min = input_roi.min()
        value_range = value_max - value_min
        
        scale = (input_roi - value_min) / value_range
        normalized_roi = scale 
        
        return normalized_roi
        
    @staticmethod
    def demean_normalization(batch):
        
        batch_mean = batch.mean()
        batch_min = batch.min()
        batch_max = batch.max()
        batch_normalized = 2. * (batch - batch_min) / (batch_max - batch_min) - 1.
        return batch_normalized
        
        
        
        
        