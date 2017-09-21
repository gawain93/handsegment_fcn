# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:14:28 2017

@author: dhri-dz
"""

import os
import cv2
from shutil import copyfile


def generate_orders(num):
    if num < 10:
        return "00" + str(num)
    if num < 100:
        return "0" + str(num)
    if num >= 100:
        return str(num)

input_path = "/home/dhri-dz/raw data"
output_path = "/home/dhri-dz/train data"
os.chdir(input_path)
folder_list = sorted(os.listdir(os.getcwd()))

for folder in folder_list:
    
    rgb_list = sorted(os.listdir(os.getcwd() + "/" + folder + "/rgbcut"))
    label_list = sorted(os.listdir(os.getcwd() + "/" + folder + "/labelcut"))
    depth_list = sorted(os.listdir(os.getcwd() + "/" + folder + "/depthcut"))
    assert len(rgb_list) == len(label_list)
    assert len(depth_list) == len(label_list)
    
    for fl in range(len(rgb_list)):
        
        rgb_num = rgb_list[fl].split('_')[-1]
        rgb_num = int(rgb_num.split('.')[0])
        label_num = label_list[fl].split('_')[-1]
        label_num = int(label_num.split('.')[0])
        depth_num = depth_list[fl].split('_')[-1]
        depth_num = int(depth_num.split('.')[0])
        assert rgb_num == label_num
        assert depth_num == label_num
        
        copyfile(os.getcwd() + "/" + folder + "/rgbcut" + "/" + rgb_list[fl], output_path + "/rgb/" + folder + '_rgb_' + generate_orders(rgb_num)+".jpg")
        copyfile(os.getcwd() + "/" + folder + "/labelcut" + "/" + label_list[fl], output_path + "/label/" + folder + '_label_' + generate_orders(label_num)+".jpg")
        copyfile(os.getcwd() + "/" + folder + "/depthcut" + "/" + depth_list[fl], output_path + "/depth/" + folder + '_depth_' + generate_orders(depth_num)+".jpg")

os.chdir(output_path)
rgb_list = sorted(os.listdir(os.getcwd()  + "/rgb"))
label_list = sorted(os.listdir(os.getcwd() + "/label"))
depth_list = sorted(os.listdir(os.getcwd() + "/depth"))

for i in range(len(rgb_list)):
    rgb = cv2.imread("./rgb/" + rgb_list[i])
    label = cv2.imread("./label/" + label_list[i])
    depth = cv2.imread("./depth/" + depth_list[i])
    
    rgb_fh = cv2.flip(rgb,0)
    label_fh = cv2.flip(label,0)
    depth_fh = cv2.flip(depth,0)
    
    cv2.imwrite("./rgb_hor/" + "fh_" + rgb_list[i] , rgb_fh)
    cv2.imwrite("./label_hor/" + "fh_" + label_list[i]  , label_fh)
    cv2.imwrite("./depth_hor/" + "fh_" + depth_list[i] , depth_fh)
    
    
    
    