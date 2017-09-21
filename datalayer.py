# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:47:47 2017

@author: dhri-dz
"""


import h5py
from keras.utils import np_utils
import numpy as np
import os
from utils import utils


class datalayer(object):
    
    def __init__(self, train_data_path):
        
        self.train_data_path = train_data_path
        os.chdir(self.train_data_path)
        
    def train_data_generator(self):
        
        while 1:
            
            for file_name in os.listdir(self.train_data_path):   # in this folder is the preprocessed  
            
                f = h5py.File(file_name, 'r')
                img_batch = f["imgs_for_training"][:]
                labels_temp = f["labels_for_training"][:]
                labels_temp = labels_temp[:,:,:,0]
                labels_batch = np.ndarray([labels_temp.shape[0], labels_temp.shape[1]*labels_temp.shape[2] ,2])
            
                utils.normalization(img_batch)
                for i in range(labels_temp.shape[0]):
                    label = labels_temp[i,:,:]                         # 256 * 256
                    label.reshape( label.shape[1]*label.shape[0], 1)   # 256² * 1
                    label = np.around(label / label.max())             # normalization 0,1
                    label = np_utils.to_categorical(label)             # 256² * 2 convert to two catagories
                    labels_batch[i,:,:] = label                        # batch_size * 256² * 2 , labels of a batch
                
                yield (img_batch, labels_batch)
    
def test():
    
    dl = datalayer('/home/dhri-dz/testfolder')
    data_generator = dl.train_data_generator()
    img_batch, labels_batch = data_generator.next()   
    

if __name__ == "__main__":
    test()
            