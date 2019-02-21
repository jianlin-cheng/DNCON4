# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2017

@author: Jie Hou
"""

from collections import defaultdict
#import cPickle as pickle
import pickle

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Lambda, add, concatenate,ConvLSTM2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras import utils
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, MaxPooling1D, AveragePooling1D,UpSampling1D, LSTM,Average
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.activations import tanh, softmax

import numpy as np


import tensorflow as tf
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D_test(filters, kernel_size, subsample,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        return Activation("sigmoid")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(filters, kernel_size, subsample,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, subsample=(1, 1), use_bias=True):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_relu1D(filters, kernel_size, subsample, use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        return Activation("relu")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_relu2D(filters, nb_row, nb_col, subsample=(1, 1), use_bias=True):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", padding="same")(input)
        # norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(conv)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(filters, kernel_size, subsample,name,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(axis=-1,name="%s_nor" % name)(conv)
        return Dense(units=3, kernel_initializer="he_normal",name="%s_softmax" % name, activation="softmax")(norm)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_sigmoid2D(filters, nb_row, nb_col, subsample=(1, 1), use_bias=True):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

def _attention_layer(input_dim):
    def f(input):
        attention_probs = Dense(input_dim, activation='softmax')(input)
        attention_mul = merge([input, attention_probs],output_shape=input_dim, mode='mul')
        return attention_mul
    return f

def _conv_bn_relu1D_drop(filters, kernel_size, subsample,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        
        norm = BatchNormalization(axis=-1)(conv)
        acti = Activation("relu")(norm)
        return Dropout(0.2)(acti)
    return f

#https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
class generatePairwiseF_keras1(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''
    def __init__(self, output_shape, **kwargs):
        super(generatePairwiseF, self).__init__(**kwargs)
        self._output_shape = tuple(output_shape)
        super(generatePairwiseF, self).__init__()
    
    def get_output_shape_for(self,input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        #shape2 = shape[1:]
        #shape2[1] *= 3
        #shape2[0] *= shape2[0]
        #return tuple(self._output_shape)
        return (input_shape[0],self._output_shape[0],self._output_shape[1],self._output_shape[2])
    
    def call(self, x, mask=None):
        new_x = x[0]
        #print "The x shape is: ", x.get_shape()
        shape =x.get_shape()  
        #output =K.zeros((shape[1]*shape[1],shape[2]*3)) # initialize matrix
        output_list  = []
        for i in range(0,shape[1]):
            for j in range(0,shape[1]):
                first = new_x[i]
                #print "first: ", first
                sec = new_x[j]
                comb =(first+sec)/2
                output_list.append(K.concatenate([first, comb,sec], axis=0))
        output = tf.stack(output_list)
        #print "The output shape is: ", output.get_shape()
        #print "The self._output_shape is: ", self._output_shape
        outputnew =  Reshape(self._output_shape)(output) 
        #outputnew =  K.reshape(output,self._output_shape)
        #print "The outputnew shape is: ", outputnew.get_shape()
        return outputnew

class generatePairwiseF_batchsize1(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''

    def __init__(self, output_shape, **kwargs):
        super(generatePairwiseF, self).__init__(**kwargs)
        self._output_shape = tuple(output_shape)
        super(generatePairwiseF, self).__init__()

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return (input_shape[0], self._output_shape[0], self._output_shape[1], self._output_shape[2])

    def call(self, x, mask=None):
        orign = x[0]
        temp = tf.fill([orign.shape[0], orign.shape[0], orign.shape[1]], 1.0)
        first = tf.multiply(temp, orign)
        second = tf.transpose(first, [1, 0, 2])
        avg = tf.div(tf.add(first, second), 2)
        output = tf.concat([first, second, avg], axis=-1)

        outputnew = K.reshape(output, (-1, self._output_shape[0], self._output_shape[1], self._output_shape[2]))
        return outputnew

class generatePairwiseF(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''
    def __init__(self, output_shape, batch_size, **kwargs):
        self._output_shape = tuple(output_shape)
        self._batch_size = batch_size
        #super(generatePairwiseF, self).__init__()
        super(generatePairwiseF, self).__init__(**kwargs)
    
    def compute_output_shape(self,input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return (input_shape[0],self._output_shape[0], self._output_shape[1], self._output_shape[2])
    
    def call(self, x, mask=None):
        dim = x.shape.as_list()
        if dim[0] is None:
            print(self._batch_size)
        else:
            self._batch_size = dim[0]
            print(self._batch_size)
        for i in range(0, self._batch_size):
            orign = x[i]
            temp = tf.fill([orign.shape[0],orign.shape[0],orign.shape[1]], 1.0)
            first = tf.multiply(temp, orign)
            second = tf.transpose(first, [1,0,2])
            avg = tf.div(tf.add(first, second), 2)
            combine = tf.concat([first, second, avg], axis=-1)
            expand = tf.reshape(combine, [1, combine.shape[0],combine.shape[1],combine.shape[2]])
            if (i==0):
                output = expand
            else:
                output = tf.concat([output, expand], axis=0)
        outputnew =  K.reshape(output,(-1,self._output_shape[0],self._output_shape[1],self._output_shape[2]))
        return outputnew
    
    def get_config(self):
        config = {'batch_size': self._batch_size,'output_shape': self._output_shape}
        base_config = super(generatePairwiseF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def convert2tensor(x):
    return K.concatenate([x])


class loadPairwiseF(Layer):
    '''
        (l*l,n) -> (l,l,n)
    '''
    def __init__(self, output_shape, **kwargs):
        super(loadPairwiseF, self).__init__(**kwargs)
        self._output_shape = tuple(output_shape)
        super(loadPairwiseF, self).__init__()
    
    def get_output_shape_for(self,input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        #shape2 = shape[1:]
        #shape2[1] *= 3
        #shape2[0] *= shape2[0]
        #return tuple(self._output_shape)
        return (input_shape[0],self._output_shape[0],self._output_shape[1],self._output_shape[2])
    
    def call(self, x, mask=None):
        new_x = x[0]
        #print "The x shape is: ", x.get_shape()
        shape =x.get_shape()  
        #print "The input shape is: ", new_x.get_shape()
        outputnew =  Reshape(self._output_shape)(new_x) 
        #outputnew =  K.reshape(new_x,self._output_shape)
        #print "The inputnew shape is: ", outputnew.get_shape()
        return outputnew

def DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    print("Setting hidden models as ",hidden_type)
    ########################################## set up 1D model
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        # DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        for i in range(0,nb_layers):
            DNCON4_1D_conv = _conv_relu1D(filters=filterss, kernel_size=fsz, subsample=1,use_bias=use_bias)(DNCON4_1D_conv)
        
        DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]  
    
    #ss_model = Model(input=DNCON4_1D_input, output=DNCON4_1D_out)
    #ss_model.summary()
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3),batch_size=batch_size)(DNCON4_1D_out)
    
    
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)   
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CNN = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    
    return DNCON4_CNN

def DNCON4_with_paras_2D(win_array,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    print("Setting hidden models as ",hidden_type)
    ########################################## set up 1D model
    # load 2D contact features
    filter_sizes=win_array
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)   
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        #DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CNN = Model(inputs=[contact_input], outputs=DNCON4_flatten)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    
    return DNCON4_CNN

def DNCON4_with_paras1(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    #RCNN
    # for fsz in filter_sizes:
    #     DNCON4_1D_conv = DNCON4_1D_input
    #     conv_l = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')
    #     DNCON4_1D_conv = conv_l(DNCON4_1D_conv)
    #     for n in range(nb_layers):
    #         DNCON4_1D_conv = RCL_block(filterss, DNCON4_1D_conv,fsz)
        
    #     DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    # if len(filter_sizes)>1:
    #     DNCON4_1D_out = Average(DNCON4_1D_convs)
    # else:
    #     DNCON4_1D_out = DNCON4_1D_convs[0]  
    # DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3),batch_size=batch_size)(DNCON4_1D_out)
    
    #RESATT
    # for fsz in filter_sizes:
    #     DNCON4_1D_conv = DNCON4_1D_input
    #     DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
    #     for i in range(0, 2):
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = Dropout(0.15)(DNCON4_1D_conv)

    #     DNCON4_1D_conv = _attention_layer(filterss)(DNCON4_1D_conv)

    #     DNCON4_1D_convs.append(DNCON4_1D_conv)

    # if len(filter_sizes) > 1:
    #     DNCON4_1D_out = Average(DNCON4_1D_convs)
    # else:
    #     DNCON4_1D_out = DNCON4_1D_convs[0]

    # DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3),batch_size=batch_size)(DNCON4_1D_out)
    

    #Frac
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        for i in range(0, 3):
            DNCON4_1D_conv = fractal_block(filters=40, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
            DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
            DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)

        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length, DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
   
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)   
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CNN = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    
    return DNCON4_CNN

def fractal_block_incepres_2D(filters, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        c1 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c2 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M1 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M1)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M1)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M2 = concatenate([c2, c3, c4], axis = -1)
        c2 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M3 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M3)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M3)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M4 = concatenate([c1, c2, c3, c4], axis = -1)
        return M4
    return f

def DNCON4_RCIncep_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):

    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

         ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_0 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filterss+32, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filterss,fsz)
            DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)       
        DNCON4_1D_convs.append(DNCON4_1D_conv) 
    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    print(DNCON4_genP.shape)
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        conv_l = Conv2D(filters=23, kernel_size=(fsz, fsz), padding='same', activation='relu')
        DNCON4_2D_conv = conv_l(DNCON4_2D_conv)
        for i in range(0,5):
            DNCON4_2D_conv = RCL_block_2D(23, DNCON4_2D_conv, fsz, fsz)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    DNCON4_New = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_New.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_New

def RCL_block(input_num_filters,l,fsz):
    out_num_filters = input_num_filters
    
    conv1 = Conv1D(filters=out_num_filters, kernel_size=fsz, padding='same')
    stack1 = conv1(l)       
    stack2 = BatchNormalization()(stack1)
    stack3 = Activation("relu")(stack2)
    
    conv2 = Conv1D(filters=out_num_filters, kernel_size=fsz, padding='same', kernel_initializer = 'he_normal')
    stack4 = conv2(stack3)
    stack5 = add([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = Activation("relu")(stack6)
    
    conv3 = Conv1D(filters=out_num_filters, kernel_size=fsz, padding='same', weights = conv2.get_weights())
    stack8 = conv3(stack7)
    stack9 = add([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = Activation("relu")(stack10)    
    
    conv4 = Conv1D(filters=out_num_filters, kernel_size=fsz, padding='same', weights = conv2.get_weights())
    stack12 = conv4(stack11)
    stack13 = add([stack1, stack12])
    stack14 = BatchNormalization(axis=-1)(stack13)
    stack15 = Activation("relu")(stack14)    
    stack16 = Dropout(0.2)(stack15)
    
    return stack16

def RCL_block_2D(input_num_filters,l,nb_row, nb_col):
    out_num_filters = input_num_filters
    
    conv1 = Conv2D(filters=out_num_filters, kernel_size=(nb_row, nb_col), padding='same')
    stack1 = conv1(l)       
    stack2 = BatchNormalization()(stack1)
    stack3 = Activation("relu")(stack2)
    
    conv2 = Conv2D(filters=out_num_filters, kernel_size=(nb_row, nb_col), padding='same', kernel_initializer = 'he_normal')
    stack4 = conv2(stack3)
    stack5 = add([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = Activation("relu")(stack6)
    
    conv3 = Conv2D(filters=out_num_filters, kernel_size=(nb_row, nb_col), padding='same', weights = conv2.get_weights())
    stack8 = conv3(stack7)
    stack9 = add([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = Activation("relu")(stack10)    
    
    conv4 = Conv2D(filters=out_num_filters, kernel_size=(nb_row, nb_col), padding='same', weights = conv2.get_weights())
    stack12 = conv4(stack11)
    stack13 = add([stack1, stack12])
    stack14 = BatchNormalization(axis=-1)(stack13)
    stack15 = Activation("relu")(stack14)    
    stack16 = Dropout(0.2)(stack15)
    
    return stack16
def DeepCovRCNN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        conv_l = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')
        DNCON4_1D_conv = conv_l(DNCON4_1D_conv)
        
        for n in range(nb_layers):
            DNCON4_1D_conv = RCL_block(filterss, DNCON4_1D_conv,fsz)
        
        DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]  

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3),batch_size=batch_size)(DNCON4_1D_out)
    
        # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    # print(DNCON4_1D_input.shape)
    # print(DNCON4_genP.shape)
    # print(contact_input.shape)
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        conv_l = Conv2D(filters=filterss, kernel_size=(fsz, fsz), padding='same', activation='relu')
        DNCON4_2D_conv = conv_l(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = RCL_block_2D(filterss, DNCON4_2D_conv, fsz, fsz)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_RCNN = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_RCNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    return DNCON4_RCNN

### 
def GRCL(inp, filterss, n_iter, fsz):
    
    conv_rec = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')
    
    conv_gate_rec = Conv1D(filters=filterss, kernel_size=1, padding='same', activation='relu')
    
    for i in xrange(n_iter):
        
        if i==0:
            #Feed forward
            conv_f = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')(inp)
            bn_f = BatchNormalization(mode=0, axis=-1)(conv_f)
            x = Activation('relu')(bn_f)
            
            #Gated
            conv_gate_f = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')(inp)
            bn_gate_f = BatchNormalization(mode=0, axis=-1)(conv_gate_f)
        
        else:
        
            c_rec = conv_rec(x)
            bn_rec = BatchNormalization(mode=0, axis=-1)(c_rec)
            
            c_gate_rec = conv_gate_rec(x)
            bn_gate_rec = BatchNormalization(mode=0, axis=-1)(c_gate_rec)
            #gate_add = Add()([bn_gate_rec, bn_gate_f])
            gate_add = merge([bn_gate_rec, bn_gate_f], mode='sum')
            gate = Activation('sigmoid')(gate_add)
            
            #gate_mul = Multiply()([bn_rec,gate])
            gate_mul = merge([bn_rec,gate], mode='mul')
            bn_gate_mul = BatchNormalization(mode=0, axis=-1)(gate_mul)
            #x_add = Add()([bn_f, bn_gate_mul])
            x_add = merge([bn_f, bn_gate_mul], mode='sum')
            
            x = Activation('relu')(x_add)
    
    return x

def DeepCovGRCNN_with_paras(win_array,feature_num,use_bias,hidden_type,filterss,nb_layers,opt):
    
    #Build Network
    filter_sizes=win_array
    DNCON4_input_shape =(None,feature_num)
    DNCON4_input = Input(shape=DNCON4_input_shape)
    DNCON4_convs = []
    for fsz in filter_sizes:
        DNCON4_conv = DNCON4_input
        conv_l = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')
        DNCON4_conv = conv_l(DNCON4_conv)
        
        for n in range(nb_layers):
            DNCON4_conv = Conv1D(filters=filterss, kernel_size=fsz, padding='same', activation='relu')(DNCON4_conv)
            DNCON4_conv = Activation('relu')(DNCON4_conv)
            DNCON4_conv = GRCL(DNCON4_conv, filterss, 5, fsz)
        
        DNCON4_conv = BatchNormalization(mode=0, axis=-1)(DNCON4_conv)
        DNCON4_conv = Activation('relu')(DNCON4_conv)   
        
        
        #need figure out NoneType
        #DNCON4_conv= Reshape((int(DNCON4_conv.shape[1]*DNCON4_conv.shape[2]),int(DNCON4_conv.shape[3])))(DNCON4_conv)
        #DNCON4_conv=Bidirectional(LSTM(512, return_sequences=True),merge_mode='sum')(DNCON4_conv)
        #DNCON4_conv=Bidirectional(LSTM(4,return_sequences=True),merge_mode='sum')(DNCON4_conv)
        #DNCON4_conv =Dense(output_dim=3, init="he_normal", activation="softmax")(DNCON4_conv)   
        
        
        DNCON4_conv = _conv_bn_softmax1D(filters=1, kernel_size=fsz, subsample=1,use_bias=use_bias,name='local_start')(DNCON4_conv)
        
        DNCON4_convs.append(DNCON4_conv)
    
    if len(filter_sizes)>1:
        DNCON4_out = Average(DNCON4_convs)
    else:
        DNCON4_out = DNCON4_convs[0]  
    DNCON4_RCNN = Model(inputs=[DNCON4_input], outputs=DNCON4_out)
    DNCON4_RCNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    
    return DNCON4_RCNN

def identity_Block_sallow(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters, kernel_size=3, subsample=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    if mode=='sum':
        x = add([x, input])
    elif mode=='concat':
        x = concatenate([x, input], axis=-1)
    return x

# def identity_Block_deep(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
#     x = _conv_relu1D(filters=filters, kernel_size=1, subsample=1,use_bias=use_bias)(input)
#     x = _conv_relu1D(filters=filters*2, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
#     x = _conv_relu1D(filters=filters, kernel_size=1, subsample=1,use_bias=use_bias)(x)
#     if with_conv_shortcut:
#         shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
#         if mode=='sum':
#             x = add([x, shortcut])
#         elif mode=='concat':
#             x = concatenate([x, shortcut], axis=-1)
#         return x
#     else:
#         if mode=='sum':
#             x = add([x, input])
#         elif mode=='concat':
#             x = concatenate([x, input], axis=-1)
#         return x

# def identity_Block_deep_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum'):
#     x = _conv_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(input)
#     x = _conv_relu2D(filters=filters*2, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(x)
#     x = _conv_relu2D(filters=filters, nb_row=1, nb_col = 1, subsample=(1, 1))(x)
#     if with_conv_shortcut:
#         shortcut = _conv_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)
#         if mode=='sum':
#             x = add([x, shortcut])
#         elif mode=='concat':
#             x = concatenate([x, shortcut], axis=-1)
#         return x
#     else:
#         if mode=='sum':
#             x = add([x, input])
#         elif mode=='concat':
#             x = concatenate([x, input], axis=-1)
#         return x

# def DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
#     DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
#     filter_sizes=win_array
#     DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
#     DNCON4_1D_convs = []
#     for fsz in filter_sizes:
#         DNCON4_1D_conv = DNCON4_1D_input
#         DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
#         DNCON4_1D_conv = _conv_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
#         # for i in range(0, nb_layers):
#         #     DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True)
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
#         DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
#         DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)
#         DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        
#         print(DNCON4_1D_conv.shape)
#         DNCON4_1D_convs.append(DNCON4_1D_conv)

#     if len(filter_sizes) > 1:
#         DNCON4_1D_out = Average(DNCON4_1D_convs)
#     else:
#         DNCON4_1D_out = DNCON4_1D_convs[0]
#     ## convert 1D to 2D
#     DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
#     print(DNCON4_genP.shape)
#     # load 2D contact features
#     contact_feature_num_2D=feature_2D_num
#     contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
#     contact_input = Input(shape=contact_input_shape)

#     ######################### now merge new data to new architecture
    
#     DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
#     DNCON4_2D_convs = []
#     for fsz in filter_sizes:
#         DNCON4_2D_conv = DNCON4_2D_input
#         DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
#         DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=1,nb_col=1,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=3,nb_col=3,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=3,nb_col=3,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_2D_conv = Dropout(0.1)(DNCON4_2D_conv)
#         DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=1,nb_col=1,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=7,nb_col=1,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=1,nb_col=7,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
#         DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=1,nb_col=1,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=7,nb_col=1,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=1,nb_col=7,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
#         DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='concat')
#         DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)
#         DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        
#         DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
#         DNCON4_2D_convs.append(DNCON4_2D_conv)

#     if len(filter_sizes) > 1:
#         DNCON4_2D_out = Average(DNCON4_2D_convs)
#     else:
#         DNCON4_2D_out = DNCON4_2D_convs[0]

    
#     DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
#     DNCON4_RES = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
#     DNCON4_RES.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

#     return DNCON4_RES
def identity_Block_deep(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters*2, kernel_size=1, subsample=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    x = _conv_relu1D(filters=filters, kernel_size=1, subsample=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def identity_Block_deep_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu2D(filters=filters*2, nb_row = 1, nb_col = 1, subsample=(1, 1), use_bias=use_bias)(input)
    x = _conv_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), use_bias=use_bias)(x)
    x = _conv_relu2D(filters=filters, nb_row=1, nb_col = 1, subsample=(1, 1))(x)
    if with_conv_shortcut:
        shortcut = _conv_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=7, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss*2, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss*4, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)     
        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    print(DNCON4_genP.shape)
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv_in = DNCON4_2D_input
        # DNCON4_2D_conv_bn = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=7, nb_col=7, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a1 = _conv_relu2D(filters=filterss, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b1 = add([DNCON4_2D_conv_a1, DNCON4_2D_conv])

        DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)

        DNCON4_2D_conv_c2 = add([DNCON4_2D_conv_a2, DNCON4_2D_conv_b2, DNCON4_2D_conv])
        
        DNCON4_2D_conv = _conv_relu2D(filters=filterss*4, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_c2)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        
        DNCON4_2D_conv_a3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv_c3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_c2)

        DNCON4_2D_conv = add([DNCON4_2D_conv_a3, DNCON4_2D_conv_b3, DNCON4_2D_conv_c3, DNCON4_2D_conv])

        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_RES = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_RES.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_RES

def block_inception_a(inputs,filterss,kernel_size,use_bias=True):
    inputs_feanum = inputs.shape.as_list()[2]

    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_1 = _conv_bn_relu1D(filters=filterss+8, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_2 = _conv_bn_relu1D(filters=filterss+16, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_3 = _conv_bn_relu1D(filters=filterss+24, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    #x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return net

def block_inception_a_2D(inputs,filters, nb_row, nb_col,subsample=(1, 1)):
    inputs_feanum = inputs.shape.as_list()[3]

    branch_0 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)

    branch_1 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)
    branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_1)

    branch_2 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)
    branch_2 = _conv_bn_relu2D(filters=filters+16, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_2)
    branch_2 = _conv_bn_relu2D(filters=filters+32, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_2)

    net = concatenate([branch_0, branch_1, branch_2], axis=-1)
    merge_branch = _conv_bn_relu2D(filters=inputs_feanum, nb_row = 1, nb_col = 1, subsample=(1, 1))(net)
    outputs = add([inputs, merge_branch])
    return outputs

def DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

         ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_0 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filterss+32, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filterss,fsz)
            DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)       
        DNCON4_1D_convs.append(DNCON4_1D_conv) 
    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    print(DNCON4_genP.shape)
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
        shape1 = DNCON4_2D_conv1.shape.as_list()[3]
        ## start inception 1
        branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        branch_1 = _conv_bn_relu2D(filters=filterss+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv2 = Dropout(0.2)(DNCON4_2D_conv2)#0.3
        DNCON4_2D_conv2 = _conv_bn_relu2D(filters=shape1, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv2)
        DNCON4_2D_conv2 = add([DNCON4_2D_conv1, DNCON4_2D_conv2])

        shape2 = DNCON4_2D_conv2.shape.as_list()[3]
        ## start inception 2
        branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_0 = _conv_bn_relu2D(filters=filterss+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_0)

        branch_1 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_1 = _conv_bn_relu2D(filters=filterss+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)
        branch_1 = _conv_bn_relu2D(filters=filterss+32, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)

        DNCON4_2D_conv3 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv3 = Dropout(0.2)(DNCON4_2D_conv3)#0.3
        DNCON4_2D_conv3 = _conv_bn_relu2D(filters=shape2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv)
        DNCON4_2D_conv3 = add([DNCON4_2D_conv2, DNCON4_2D_conv3])
        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_2D_conv = block_inception_a_2D(DNCON4_2D_conv,filterss,nb_row=fsz, nb_col=fsz)
            DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)#0.3
            
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    DNCON4_INCEP = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_INCEP.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    # DNCON4_INCEP.summary()
    return DNCON4_INCEP

def block_incepres_a(input,filterss,kernel_size,use_bias=True):

    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)

    branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)

    branch_2 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_2 = _conv_bn_relu1D(filters=filterss*2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_2)
    branch_2 = _conv_bn_relu1D(filters=filterss*4, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_2)

    # branch = merge([branch_0, branch_1, branch_2], mode='concat')

    net = merge([input, branch_0, branch_1, branch_2], mode='concat')
    return net

def block_incepres_reduc_a(input,filterss,kernel_size,use_bias=True):
    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(input)

    branch_1 = _conv_bn_relu1D(filters=filterss/4, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_1 = _conv_bn_relu1D(filters=filterss/2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)

    # branch = merge([branch_0, branch_1, branch_2], mode='concat')

    net = merge([branch_0, branch_1], mode='concat')
    return net

def block_incepres_b(input,filterss,kernel_size,use_bias=True):
    branch_0 = _conv_bn_relu1D(filters=filterss*4, kernel_size=1, subsample=1, use_bias=use_bias)(input)

    branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=1, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*4, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*4, kernel_size=1, subsample=1, use_bias=use_bias)(branch_1)

    net = merge([input, branch_0, branch_1], mode='concat')
    return net

def block_incepres_reduc_b(input,filterss,kernel_size,use_bias=True):
    branch_0 = _conv_bn_relu1D(filters=filterss/4, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_0)

    branch_1 = _conv_bn_relu1D(filters=filterss/4, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_1 = _conv_bn_relu1D(filters=filterss/2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)

    branch_2 = _conv_bn_relu1D(filters=filterss/4, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_2 = _conv_bn_relu1D(filters=filterss/2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_2)
    branch_2 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_2)

    # branch = merge([branch_0, branch_1, branch_2], mode='concat')

    net = merge([branch_0, branch_1, branch_2], mode='concat')
    return net

def block_incepres_c(input,filterss,kernel_size,use_bias=True):
    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)

    branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(input)
    branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=1, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*4, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)
    branch_1 = _conv_bn_relu1D(filters=filterss*4, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(branch_1)

    # branch = merge([branch_0, branch_1, branch_1], mode='concat')

    net = merge([input, branch_0, branch_1], mode='concat')
    return net

def DeepIncepRes1D_with_paras(win_array, feature_num, use_bias, hidden_type, filterss, nb_layers, opt):
    filter_sizes = win_array
    DNCON4_input_shape = (None, feature_num)
    DNCON4_input = Input(shape=DNCON4_input_shape)

    DNCON4_convs = []
    for fsz in filter_sizes:
        net = DNCON4_input
        net = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(net)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(net)
        branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=fsz, subsample=1, use_bias=use_bias)(net)
        net = merge([branch_0, branch_1], mode='concat')
        # net = Dropout(0.3)(net)

        ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(net)
        branch_0 = _conv_bn_relu1D(filters=filterss*2, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(net)
        branch_1 = _conv_bn_relu1D(filters=filterss*2, kernel_size=1, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filterss*4, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        net = merge([branch_0, branch_1], mode='concat')
        # net = Dropout(0.3)(net)

        for idx in range(nb_layers):
            net = block_incepres_a(net,filterss,fsz)
        # block_incepres_reduc_a(net,filterss,fsz)

        # for idx in range(nb_layers+2):
        #     net = block_incepres_b(net,filterss,fsz)
        # block_incepres_reduc_b(net,filterss,fsz)

        # for idx in range(nb_layers):
        #     net = block_incepres_c(net,filterss,fsz)
        DNCON4_conv = _conv_bn_softmax1D(filters=1, kernel_size=fsz, subsample=1, use_bias=use_bias,
                                             name='local_start')(net)
        DNCON4_convs.append(DNCON4_conv)

    if len(filter_sizes) > 1:
        DNCON4_out = Average(DNCON4_convs)
    else:
        DNCON4_out = DNCON4_convs[0]

    DNCON4_CNN = Model(inputs=[DNCON4_input], outputs=DNCON4_out)
    DNCON4_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_CNN

def identity_Block(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def identity_Block_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False, mode='sum'):
    x = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(input)
    x = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(x)
    if with_conv_shortcut:
        shortcut = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x
#attention_vec
def DeepCovResAtt_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        for i in range(0, nb_layers):
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = Dropout(0.15)(DNCON4_1D_conv)

        DNCON4_1D_conv = _attention_layer(filterss)(DNCON4_1D_conv)

        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3),batch_size=batch_size)(DNCON4_1D_out)
    
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
        for i in range(0, nb_layers):
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = Dropout(0.15)(DNCON4_2D_conv)

        DNCON4_2D_conv = _attention_layer(filterss)(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CNN = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_CNN

def identity_Block_CRMN(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_bn_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
    x = _conv_bn_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_bn_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def identity_Block_CRMN_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False, mode='sum'):
    x = _conv_bn_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(input)
    x = _conv_bn_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(x)
    if with_conv_shortcut:
        shortcut = _conv_bn_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=(1,1))(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        for i in range(0, nb_layers):
            cnn = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
            res = identity_Block_CRMN(cnn, filters=filterss, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            cnnres = add([cnn, res])
            cnnres = Dropout(0.2)(cnnres)

            lstm = LSTM(filterss, return_sequences=True)(cnnres)
            Lstmlayer = LSTM(filterss, return_sequences=True)(lstm)
            DNCON4_1D_conv = cnnres

        DNCON4_1D_conv = concatenate([DNCON4_1D_conv, Lstmlayer], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length, DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        for i in range(0, nb_layers):
            cnn = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
            res = identity_Block_CRMN_2D(cnn, filters=filterss, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            cnnres = add([cnn, res])
            cnnres = Dropout(0.2)(cnnres)
            print(cnnres.shape)
            trans2lstm = Reshape((5, cnnres.shape.as_list()[1], cnnres.shape.as_list()[1], -1))(cnnres)
            lstm = ConvLSTM2D(filters=filterss, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(trans2lstm)
            Lstmlayer = ConvLSTM2D(filters=filterss, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(lstm)
            DNCON4_2D_conv = cnnres

        LastLstm = ConvLSTM2D(filters=filterss, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(Lstmlayer)
        DNCON4_2D_conv = concatenate([DNCON4_2D_conv, LastLstm], axis=-1)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CRMN = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_CRMN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_CRMN

def fractal_block(filters, kernel_size, subsample=1, use_bias = True):
    def f(input):
        c1 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(input)
        c2 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(input)
        c3 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(input)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(input)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(c4)
        M1 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M1)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M1)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(c4)
        M2 = concatenate([c2, c3, c4], axis = -1)
        c2 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M2)
        c3 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M2)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M2)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(c4)
        M3 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M3)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(M3)
        c4 = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=subsample, use_bias=use_bias)(c4)
        M4 = concatenate([c1, c2, c3, c4], axis = -1)
        return M4
    return f

def fractal_block_2D(filters, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        c1 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c2 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(input)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M1 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M1)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M1)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M2 = concatenate([c2, c3, c4], axis = -1)
        c2 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M2)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M3 = concatenate([c3, c4], axis = -1)
        c3 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M3)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(M3)
        c4 = _conv_relu2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample)(c4)
        M4 = concatenate([c1, c2, c3, c4], axis = -1)
        return M4
    return f

def DeepFracNet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        for i in range(0, nb_layers):
            DNCON4_1D_conv = fractal_block(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
            DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
            DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        # DNCON4_1D_conv = _conv_bn_softmax1D(filters=1, kernel_size=fsz, subsample=1, use_bias=use_bias, name='local_start')(DNCON4_1D_conv)
        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length, DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        for i in range(0, nb_layers):
            DNCON4_2D_conv = fractal_block_2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
            DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
            DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)

        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_FRAC = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_FRAC.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_FRAC

#def block_inception_a(input,filterss,kernel_size,use_bias=True):

#     branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(input)

#     branch_1 = _conv_bn_relu1D(filters=filterss+8, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(input)

#     branch_2 = _conv_bn_relu1D(filters=filterss+16, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(input)

#     branch_3 = _conv_bn_relu1D(filters=filterss+24, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(input)

#     #x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
#     net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
#     return net

# def block_inception_a_2D(input,filters, nb_row, nb_col,subsample=(1, 1)):

#     branch_0 = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)

#     branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)

#     branch_2 = _conv_bn_relu2D(filters=filters+16, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)

#     branch_3 = _conv_bn_relu2D(filters=filters+24, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)
#     net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
#     return net

# def DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
#     DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
#     filter_sizes=win_array
#     DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
#     DNCON4_1D_convs = []
#     for fsz in filter_sizes:
#         DNCON4_1D_conv = DNCON4_1D_input
#         DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

#         ## start inception 1
#         branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
#         branch_1 = _conv_bn_relu1D(filters=filterss+8, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
#         DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
#         DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

#         ## start inception 2
#         branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
#         branch_0 = _conv_bn_relu1D(filters=filterss+8, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

#         branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
#         branch_1 = _conv_bn_relu1D(filters=filterss+8, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
#         branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

#         DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
#         DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

#         # 35 x 35 x 384
#         # 4 x Inception-A blocks
#         for idx in range(nb_layers):
#             DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filterss,fsz)
#             DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3
            
#         DNCON4_1D_convs.append(DNCON4_1D_conv)

#     if len(filter_sizes) > 1:
#         DNCON4_1D_out = Average(DNCON4_1D_convs)
#     else:
#         DNCON4_1D_out = DNCON4_1D_convs[0]

#     DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
#     print(DNCON4_genP.shape)
#     # load 2D contact features
#     contact_feature_num_2D=feature_2D_num
#     contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
#     contact_input = Input(shape=contact_input_shape)

#     ######################### now merge new data to new architecture
    
#     DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
#     DNCON4_2D_convs = []
#     for fsz in filter_sizes:
#         DNCON4_2D_conv = DNCON4_2D_input
#         DNCON4_2D_conv = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)

#         ## start inception 1
#         branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
#         branch_1 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
#         DNCON4_2D_conv = concatenate([branch_0, branch_1], axis=-1)
#         DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)#0.3

#         ## start inception 2
#         branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
#         branch_0 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_0)

#         branch_1 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
#         branch_1 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)
#         branch_1 = _conv_bn_relu2D(filters=filterss*4, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)

#         DNCON4_2D_conv = concatenate([branch_0, branch_1], axis=-1)
#         DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)#0.3

#         # 35 x 35 x 384
#         # 4 x Inception-A blocks
#         for idx in range(nb_layers):
#             DNCON4_2D_conv = block_inception_a_2D(DNCON4_2D_conv,filterss,nb_row=fsz, nb_col=fsz)
#             DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)#0.3
            
#         DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
#         DNCON4_2D_convs.append(DNCON4_2D_conv)

#     if len(filter_sizes) > 1:
#         DNCON4_2D_out = Average(DNCON4_2D_convs)
#     else:
#         DNCON4_2D_out = DNCON4_2D_convs[0]

#     DNCON4_flatten = Flatten()(DNCON4_2D_out)
#     DNCON4_INCEP = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
#     DNCON4_INCEP.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

#     return DNCON4_INCEP
##################For test

#win_array=[6]
#filter_sizes=win_array
#feature_num=8
#DNCON4_input_shape =(None,feature_num)
#DNCON4_input = Input(shape=DNCON4_input_shape)
#DNCON4_convs = []
#DNCON4_conv = DNCON4_input
#fsz=6
#l_settings = conv_l


#DNCON4_CNN = DeepCovRCNN_with_paras([6],20,True,'relu',10,5,'adam')

#DNCON4_CNN = DeepCovGRCNN_with_paras([6],20,True,'relu',10,5,'adam')
#DNCON4_CNN = DeepResnet1D_with_paras([6],20,True,'relu',10,5,'adam')
#DNCON4_CNN = DeepInception1D_with_paras([6],20,True,'relu',10,5,'adam')
#DNCON4_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='adam')
#X = np.random.rand(60,100,20)
#Y = np.random.randint(0,2,60*100)
#Y = Y.reshape(60,100,1)
#test_targets = np.zeros((60,100, 3 ), dtype=int)
#for i in range(0, Y.shape[0]):
#    for j in range(0, Y.shape[1]):
#        test_targets[i][j][Y[i][j]] = 1

#DNCON4_CNN.fit([X], test_targets, batch_size=10,nb_epoch=5, verbose=1)




"""  Test example for 1D and 2D network

Alldata_1D= np.random.sample((2,10,26))
Alldata_2D= np.random.sample((2,10,10,3))
Alltarget_1D= np.random.randint(0,2,10*10*2)
Alltarget_1D= Alltarget_1D.reshape(2,10*10)

#model_ResCNN.fit(Alldata, Alltarget, batch_size=1,nb_epoch=20, validation_split=0.1, verbose=1) # 10%  as validation 
#model_ResCNN.fit([Alldata_1D,Alldata_2D], Alltarget_1D, batch_size=1,nb_epoch=20,  validation_data=(Alldata_1D, Alltarget_1D), verbose=1)


sequence_length=10
ss_feature_num = 3
sa_feature_num = 2
contact_feature_num = 3
pssm_feature_num = 21


########################################## set up 1D model
ss_input_shape =(sequence_length,ss_feature_num+sa_feature_num+pssm_feature_num)
filterss = 5
filter_size = 6
nb_layers=5
DNCON4_input = Input(shape=ss_input_shape)
DNCON4_convs = []

DNCON4_conv = DNCON4_input
#DNCON4_conv = BatchNormalization(mode=0, axis=-1)(DNCON4_conv)
for i in range(0,nb_layers):
    DNCON4_conv = _conv_bn_relu1D_test(filters=filterss, kernel_size=filter_size, subsample=1,use_bias=True)(DNCON4_conv)

# test model
ss_model = Model(input=DNCON4_input, output=DNCON4_conv)
ss_model.summary()
ss_model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="sgd")
#ss_model.fit(Alldata_1D, Alltarget_1D, batch_size=1,nb_epoch=20, verbose=1)


check_feature1 = ss_model.predict([Alldata_1D])

#https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
class generatePairwiseF(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''
    
    def __init__(self, output_shape, **kwargs):
        super(generatePairwiseF, self).__init__(**kwargs)
        self._output_shape = tuple(output_shape)
        super(generatePairwiseF, self).__init__()
    
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return (input_shape[0], self._output_shape[0], self._output_shape[1], self._output_shape[2])
    
    def call(self, x, mask=None):
        orign = x[0]
        temp = tf.fill([orign.shape[0], orign.shape[0], orign.shape[1]], 1.0)
        first = tf.multiply(temp, orign)
        second = tf.transpose(first, [1, 0, 2])
        avg = tf.div(tf.add(first, second), 2)
        output = tf.concat([first, second, avg], axis=-1)
        
        outputnew = K.reshape(output, (-1, self._output_shape[0], self._output_shape[1], self._output_shape[2]))
        return outputnew

ss_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filterss*3))(DNCON4_conv)


#test model
model = Model(input=DNCON4_input, output=ss_genP)
model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="sgd")
model.summary()

check_feature2 = model.predict([Alldata_1D])
#check_feature2[0][:,:,0]
#model.fit(Alldata_1D, Alltarget_1D, batch_size=1,nb_epoch=20, verbose=1)


def convert2tensor(x):
    return K.concatenate([x])

# load 2D contact features
contact_feature_num_2D=3
contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
contact_input = Input(shape=contact_input_shape)
#contact_c2t = Lambda(convert2tensor, output_shape=(sequence_length*sequence_length,contact_feature_num_2D), input_shape=(sequence_length*sequence_length,contact_feature_num_2D))(contact_input)

#contact_loadP=loadPairwiseF(output_shape=(sequence_length,sequence_length,contact_feature_num_2D))(contact_c2t)




######################### now merge new data to new architecture

#DNCON4_combine_feature = merge([ss_genP,contact_loadP], mode='concat', concat_axis=-1) 
DNCON4_combine_feature = merge([ss_genP,contact_input], mode='concat', concat_axis=-1) 
# shape=(1, 40, 40, 240) = (1, 40, 40, 3*60[1D] + 60[2D])
# test model
model = Model(input=[DNCON4_input,contact_input], output=[DNCON4_combine_feature])
model.summary()
#from keras.utils.visualize_util import plot
#plot(model, to_file='/Users/jiehou/Mizzou/PHD_projects/Residual_NN_development/Jie_implement/DNCON4_architecture.png')


def Kreshape(x,shape): 
    #xnew=Reshape(shape)(x)
    xnew=K.reshape(x,shape)
    #return  K.concatenate([xnew])
    return  xnew

### set 2D residual neural network
#DNCON4_combine_feature_new = Lambda(Kreshape,arguments={'shape':(1,sequence_length, sequence_length, 3*filterss+3)})(DNCON4_combine_feature)

# test model
model = Model(input=[DNCON4_input,contact_input], output=[DNCON4_combine_feature])
model.summary()
#plot(model, to_file='/Users/jiehou/Mizzou/PHD_projects/Residual_NN_development/Jie_implement/DNCON4_architecture.png')


# Helper to build a conv -> BN -> relu block
def _conv_relu(filters, kernel_size, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, (kernel_size, nb_col), subsample=subsample,
                             init="he_normal", padding="same",dim_ordering="tf")(input)
        #norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(conv)
    
    return f

DNCON4_conv2 = _conv_relu(filters=15, kernel_size=3, nb_col=3, subsample=(1, 1))(DNCON4_combine_feature)
DNCON4_conv2 = _conv_relu(filters=1, kernel_size=3, nb_col=3, subsample=(1, 1))(DNCON4_conv2)

    
     
# test model
model = Model(input=[DNCON4_input,contact_input], output=[DNCON4_conv2])
model.summary()
#plot(model, to_file='/Users/jiehou/Mizzou/PHD_projects/Residual_NN_development/Jie_implement/DNCON4_architecture.png')

# Classifier block
DNCON4_flatten1 = Flatten()(DNCON4_conv2)
#last_layer = Lambda(Kreshape,arguments={'shape':(40*40, 1)})(flatten1)
DNCON4_model = Model(input=[DNCON4_input,contact_input], output=[DNCON4_flatten1])
DNCON4_model.summary()

DNCON4_model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer="sgd")
DNCON4_model.fit([Alldata_1D,Alldata_2D], Alltarget_1D, batch_size=1,nb_epoch=20, verbose=1)
"""
