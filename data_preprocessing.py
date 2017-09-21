# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:41:42 2017

@author: Jiawen Xu
"""

import cv2
import os
import h5py
import numpy as np
from utils import utils
from random import shuffle

class data_preprocessing(object):
    
    def __init__(self, input_path, output_path, batch_size, object_size):
        
        os.chdir(input_path)
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.object_size = object_size
        self.rgb_list = sorted(os.listdir(os.getcwd() + "/rgbmap"))
        self.label_list = sorted(os.listdir(os.getcwd() + "/mask"))
        self.depth_list = sorted(os.listdir(os.getcwd() + "/depthmap"))
        self.list = [i for i in range(len(self.rgb_list))]
        shuffle(self.list)
        print type(self.rgb_list[1])
        
    def processing(self):
        
        num_step = ceil (len(self.list) / self.batch_size) 
        batch_counter = 0
        batch_num = 0
        frame_index = 0
        img_channels = 3
        uti = utils(self.object_size, self.input_path)
        assert len(uti.hand_middle_x) == len(self.list)
        
        while frame_index < len(self.list):
            
            rgb = cv2.imread(self.input_path + '/rgbmap/' + self.rgb_list[frame_index] , 1)
            depth = cv2.imread(self.input_path + '/depthmap/'+ self.depth_list[frame_index] , 1)  
            label = cv2.imread(self.input_path + '/mask/' + self.label_list[frame_index] , 1)
            
            rgb_roi = uti.cut_ROI(rgb, frame_index)
            label_roi = uti.generate_label(label, frame_index)
            depth_roi = uti.cut_ROI(depth, frame_index)  
            '''
            labels_for_training = np.ndarray([self.batch_size, self.object_size[0]*2 , self.object_size[1]*2, img_channels]) 
            imgs_for_training = np.ndarray([self.batch_size, self.object_size[0]*2 , self.object_size[1]*2, img_channels]) 
            depth_for_training = np.ndarray([self.batch_size, self.object_size[0]*2 , self.object_size[1]*2, img_channels])
            
            while batch_counter < self.batch_size and frame_index < len(self.list):
                
                #print frame_index
                #print batch_counter
                rgb = cv2.imread(self.input_path + '/rgbmap/' + self.rgb_list[frame_index] , 1)
                depth = cv2.imread(self.input_path + '/depthmap/'+ self.depth_list[frame_index] , 1)  
                label = cv2.imread(self.input_path + '/mask/' + self.label_list[frame_index] , 1)
                
                rgb_roi = uti.cut_ROI(rgb, frame_index)
                label_roi = uti.generate_label(label, frame_index)
                depth_roi = uti.cut_ROI(depth, frame_index)  
                #cv2.imshow('image', label_roi)
                #cv2.waitKey(0)
                
                labels_for_training[batch_counter, : , :, :] = label_roi
                depth_for_training[batch_counter, : , :, :] = depth_roi
                imgs_for_training[batch_counter, : , :, :] = rgb_roi
                
                frame_index += 1
                batch_counter += 1
                
                #if batch_counter >= self.batch_size:
            '''
            #imgs_for_training = imgs_for_training.astype(int)
            #depth_for_training = depth_for_training.astype(int)
            #labels_for_training = labels_for_training.astype(int)
            #print imgs_for_training[10, : , :, :].min()
            #cv2.imshow('image', imgs_for_training[0, : , :, :])
            #cv2.waitKey(0)
            '''
            dump_file = h5py.File(self.output_path + '/' + str(batch_num) + '.h5', 'w')
            dump_file.create_dataset("imgs_for_training", data = imgs_for_training)
            dump_file.create_dataset("depth_for_training", data = depth_for_training)
            dump_file.create_dataset("labels_for_training", data = labels_for_training)
            dump_file.close()
                    
            batch_num += 1
            batch_counter = 0
            '''
            cv2.imwrite("./rgbcut/" + self.rgb_list[frame_index], rgb_roi)
            cv2.imwrite("./labelcut/" + self.label_list[frame_index], label_roi)
            cv2.imwrite("./depthcut/" + self.depth_list[frame_index], depth_roi)
            frame_index += 1

def test():
     
    input_path = '/home/dhri-dz/raw data/2017-09-04-19-53-08_new'
    output_path = '/home/dhri-dz/raw data/2017-09-04-19-53-08_new'
    batch_size = 20
    object_size = (128,128)
    dp = data_preprocessing( input_path, output_path, batch_size, object_size)
    
    dp.processing()
    
    
if __name__ == "__main__":
    test()