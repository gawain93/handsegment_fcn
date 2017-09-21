# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:39:55 2017

@author: dhri-dz
"""

import os


data_path = "/home/dhri-dz/savedata"
os.chdir(data_path)

def convert_num(index):
    
    if index < 10:
        index = '000' + str(index)
        return index
        
    if index < 100:
        index = '00' + str(index)
        return index
        
    if index < 1000:
        index = '0' + str(index)
        return index
        
    if index >= 1000:
        index = str(index)
        return index

# change file names in the subfolder rgbmap
os.chdir(data_path + '/rgbmap')
datalist = os.listdir(os.getcwd())
for image_name in datalist:
    
    name = image_name.split('_')[0]
    index = int((image_name.split('_')[1]).split('.')[0])
    index = convert_num(index)
    
    new_filename = name + '_' + index + '.jpg'
    os.rename(image_name, new_filename)
    
    
os.chdir(data_path + '/depthmap')
datalist = os.listdir(os.getcwd())
for image_name in datalist:
    
    name = image_name.split('_')[0]
    index = int((image_name.split('_')[1]).split('.')[0])
    index = convert_num(index)
    
    new_filename = name + '_' + index + '.jpg'
    os.rename(image_name, new_filename)
    
    
os.chdir(data_path + '/mask')
datalist = os.listdir(os.getcwd())
for image_name in datalist:
    
    name = image_name.split('_')[0]
    index = int((image_name.split('_')[1]).split('.')[0])
    index = convert_num(index)
    
    new_filename = name + '_' + index + '.jpg'
    os.rename(image_name, new_filename)