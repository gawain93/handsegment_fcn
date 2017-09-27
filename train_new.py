# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:04:10 2017

@author: dhri-dz
"""

from net_new import fcn32
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
from utils import utils
import cv2
import os
import glob
from input_pre import input_pre
from keras.models import model_from_json
from keras.models import model_from_yaml


file_list = sorted(os.listdir(os.getcwd()))
data_path = "/home/dhri-dz/handsegment/train data"
model_path = '/home/dhri-dz/handsegment/train data/my_model_950.h5'
img_height = 256
img_width = 256
img_channels = 3
no_class = 2
ip = input_pre(data_path, img_channels)

def sort_data(data_path):
    
    os.chdir(data_path)
    img_list = sorted(os.listdir(data_path + "/rgb"))
    label_list = sorted(os.listdir(data_path + "/label"))
    # following to make the input rgb images and the labels apart
  
    return img_list, label_list
    

def Data_Generator(batch_size, img_height, img_width, img_channels, img_list, label_list, no_class):
    
    cards = img_list[1:3000]
    np.random.shuffle(cards)
    
    while 1:
        i = 0
        img_batch = np.ndarray([batch_size, img_height, img_width, img_channels])
        label_batch = np.ndarray([batch_size, img_height, img_width, no_class])
        while i < batch_size:
            if len(cards) == 0:
               cards = img_list[1:3000]
               np.random.shuffle(cards)
               
            img = cv2.imread("./rgb/" + cards[0],1)
            video_name = cards[0].split('.')[0].split('_')[0]
            frame_num = cards[0].split('.')[0].split('_')[2]
            # normalize the image
            
            img_batch[i,...] = img
            
            label = cv2.imread('./label/' + video_name + '_label_' + frame_num + '.jpg', 1)
            label = np.around(label/255.)
            
            label = label[...,1]            
            label_batch[i,:,:,0] = label
            label_batch[i,:,:,1] = 1 - label
            i+=1
            del cards[0]
            
        img_batch = ip.global_normalization_rgb(img_batch)  
        yield (img_batch, label_batch)
        
        
def vali_generator(vali_size, img_height, img_width, img_channels, img_list, no_class):
    
    vali_batch = np.ndarray([vali_size, img_height, img_width, img_channels])
    vali_label = np.ndarray([vali_size, img_height, img_width, no_class])
   
    for i in range(vali_size):
        
        img2 = cv2.imread("./rgb/" + img_list[3001+i],1)
        vali_batch[i,...] = img2
        video_name = img_list[3001 + i].split('.')[0].split('_')[0]
        frame_num = img_list[3001 + i].split('.')[0].split('_')[2]
        
        label2 = cv2.imread('./label/' + video_name + '_label_' + frame_num + '.jpg', 1)
        label2 = np.around(label2 / 255.)
        label2 = label2[...,1]            
        vali_label[i,:,:,0] = label2
        vali_label[i,:,:,1] = 1 - label2
    
    vali_batch = ip.global_normalization_rgb(vali_batch)  
    return vali_batch, vali_label
            

def train():
    
    data_path = "/home/dhri-dz/handsegment/train data"
    img_height = 256
    img_width = 256
    img_channels = 3
    no_class = 2
    batch_size = 10
    
    model = fcn32(num_class = 2)
    model.load_weights(model_path)
    #model = load_model(model_path)
    print model.input.shape
    
    img_list, label_list = sort_data(data_path)
    vali_batch, vali_label = vali_generator(10, 256, 256, 3, img_list, 2)
    
    data_generator = Data_Generator(batch_size, img_height, img_width, img_channels, img_list, label_list, no_class)
    history = model.fit_generator(data_generator, steps_per_epoch =len(img_list)//batch_size+1, epochs = 550, validation_data=(vali_batch,vali_label))    
  
    # test the model , for debugging
    test_list = sorted(os.listdir(data_path + '/rgb'))[3100:3100+10]
    test_batch = np.ndarray([10, img_height, img_width, img_channels])
    test_label = np.ndarray([10, img_height, img_width, no_class])
    for i in range(10):       
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
    predicted = []
    predicted = model.predict(test_batch)
    return history, predicted, test_label
    
    
def test():
    data_path = "/home/dhri-dz/handsegment/train data"
    
    #img_list, label_list = sort_data(data_path)
    #data_generator = Data_Generator(batch_size, img_height, img_width, img_channels, img_list, label_list, no_class)
    history, predicted, test_label = train()
    
    history.model.save_weights('my_model_1500.h5')
    #newmodel = model_from_yaml(history.model.all.to_yaml()) 
    '''
    model_json = history.model.to_json()
    with open("/home/dhri-dz/train data/model", 'w') as json_file:
         json_file.write(model_json)
    print(history.model.to_json(indent=4))  
    '''
      
    return history, predicted, test_label
    
if __name__ == "__main__":
    history, predicted, test_label = test()

