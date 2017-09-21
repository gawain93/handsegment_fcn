# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:09:24 2017

@author: dhri-dz
"""

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Dropout, add, Activation
from keras import regularizers
from keras import optimizers
from keras.initializers import Constant, Zeros
from keras.utils import plot_model
from PIL import Image
import numpy as np


num_class = 2
weight_decay = 0.1

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor -1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    
    
def bilinear_upsample_weights(up_factor, number_of_classes):
    filter_size = up_factor * 2 - up_factor % 2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes), dtype = np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in xrange(number_of_classes):
        weights[:,:,i,i] = upsample_kernel
    return weights.astype('float32')

def fcn32(num_class, weight_decay = 0.1):
    inputs = Input(shape = (256,256,3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    conv6_1 = Conv2D(filters = 4096, kernel_size = (7,7), padding = 'same',  activation = 'relu', name='conv6_1' , kernel_regularizer=regularizers.l2(weight_decay))(vgg16.output)    #
    conv6_drop = Dropout(0.5, name='conv6_drop')(conv6_1)
    conv7_1 = Conv2D(filters = 4096, kernel_size = (1,1), padding = 'same',  activation = 'relu', name='conv7_1',  kernel_regularizer=regularizers.l2(weight_decay))(conv6_drop)     #
    conv7_drop = Dropout(0.5, name='conv7_drop')(conv7_1)
    
    # upsample the output of layer7
    score_fr = Conv2D(filters = num_class, kernel_size = (1,1), border_mode = 'same', name = 'score_fr', kernel_initializer = Zeros())(conv7_drop)
    up_score7 = Conv2DTranspose(num_class, padding="same",  strides=(2, 2), kernel_size = (4,4), name = 'up_score7',kernel_initializer =Constant(bilinear_upsample_weights(2, num_class)))(score_fr)  # ,kernel_initializer = Constant(bilinear_upsample_weights(2, num_class))
    
    # 1*1 kernel convolution of out layer 4
    score_pool4 = Conv2D(filters = num_class, kernel_size = (1,1), border_mode = 'same',name = 'score_pool4' , kernel_initializer = Zeros())(vgg16.layers[14].output)  #
    
    # fuse upsampled layer 7 and layer4
    score1 = add([up_score7, score_pool4], name = 'score1')
    # upsample the fuse
    up_score1 = Conv2DTranspose(num_class, padding="same",  strides=(2, 2), kernel_size = (4,4), name = 'up_score1', kernel_initializer = Constant(bilinear_upsample_weights(2, num_class)))(score1)   # , kernel_initializer = Constant(bilinear_upsample_weights(2, num_class))
    
    # 1 * 1 convolution of out layer 3
    score_pool3 = Conv2D(filters = num_class, kernel_size = (1,1), border_mode = 'same', name = 'score_pool3'  , kernel_initializer = Zeros())(vgg16.layers[10].output)   #'he_uniform'
    
    # fuse upsampled score1 and laver3
    score2 = add([up_score1,  score_pool3], name = 'score2')
    # upsample score2
    outputs = Conv2DTranspose(num_class, padding="same", strides =(8, 8), kernel_size = (16, 16),name = 'outputs', kernel_initializer = Constant(bilinear_upsample_weights(8, num_class)))(score2)       #, kernel_initializer = Constant(bilinear_upsample_weights(8, num_class))
    outputs = Activation(activation='softmax', name='class_out')(outputs)    
    
    model = Model(inputs, outputs)
    
    for layer in model.layers[:14]:
        layer.trainable = False
    model.layers[28].trainable = False
    plot_model(model, to_file='model.png')
    
    sgd = optimizers.SGD(lr=3e-7, decay=5e-4, momentum=0.99, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy']) 
    
    return model