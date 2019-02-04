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
from keras.layers import Input, Dense, Reshape, Activation, Flatten, Embedding, merge, Merge, Dropout, Lambda, add, concatenate,ConvLSTM2D,LSTM,Average
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm

from keras import utils
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization
from keras.activations import tanh, softmax
from keras import metrics
import numpy as np

import tensorflow as tf
from maxout import MaxoutConv2D, max_out
from maxout_test import MaxoutConv2D_Test
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D_test(filters, kernel_size, subsample,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        return Activation("sigmoid")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(filters, kernel_size, subsample,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, activation='relu', padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, subsample=(1, 1), use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1, momentum=0.99, scale=False)(conv)
        return Activation("relu")(norm)
    return f

def _conv_relu1D(filters, kernel_size, subsample, use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, activation='relu', padding="same")(input)
        return Activation("relu")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_relu2D(filters, nb_row, nb_col, subsample=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample, use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        # norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(conv)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(filters, kernel_size, subsample,name,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=subsample,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, activation='relu', padding="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(axis=-1,name="%s_nor" % name)(conv)
        return Dense(units=3, kernel_initializer=kernel_initializer,name="%s_softmax" % name, activation="softmax")(norm)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_sigmoid2D(filters, nb_row, nb_col, subsample=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=subsample,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

def _attention_layer(input_dim):
    def f(input):
        attention_probs = Dense(input_dim, activation='softmax')(input)
        attention_mul = merge([input, attention_probs],output_shape=input_dim, mode='mul')
        return attention_mul
    return f


def _weighted_mean_squared_error(y_ture, y_pred):
    y_mean = K.mean(y_ture)
    weights = 1/(1 + K.square(y_pred/y_mean))
    return K.mean(K.square((y_pred-y_ture)*weights))

def _weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def _weighted_binary_crossentropy(pos_weight=1, neg_weight=1):

    def loss(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weights = y_true * pos_weight + (1. - y_true) * neg_weight

        weighted_binary_crossentropy_vector = weights * binary_crossentropy

        return K.mean(weighted_binary_crossentropy_vector)
    return loss

def _weighted_binary_crossentropy_shield(pos_weight=1, neg_weight=1, shield=0):

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        # cross-entropy loss with weighting
        out = -(y_true * K.log(y_pred)*pos_weight+ (1.0 - y_true) * K.log(1.0 - y_pred)*neg_weight)
        return K.mean(out, axis=-1)
    return loss
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

def DNCON4_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
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
            DNCON4_1D_conv = _conv_relu1D(filters=filters, kernel_size=fsz, subsample=1,use_bias=use_bias)(DNCON4_1D_conv)
        
        DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]  
    
    #ss_model = Model(input=DNCON4_1D_input, output=DNCON4_1D_out)
    #ss_model.summary()
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filters*3),batch_size=batch_size)(DNCON4_1D_out)
    
    
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
            DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        
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

def DeepConv_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "he_normal", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0, activation = 'relu'):
    filter_sizes=win_array
        # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv = Dense(64)(DNCON4_2D_conv)
        DNCON4_2D_conv = MaxoutConv2D_Test(kernel_size=(1,1), output_dim=64, nb_features = 4, activation = activation)(DNCON4_2D_conv)
        # DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64)(DNCON4_2D_conv)

        #all param 645473
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        # DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        # DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters*4, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        
        #all param 1025985
        # for i in range(0,nb_layers):
        #     DNCON4_2D_conv = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        #     DNCON4_2D_conv = Dropout(0.1)(DNCON4_2D_conv)

        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    # DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_CNN = Model(inputs=[contact_input], outputs=DNCON4_2D_out)
    DNCON4_CNN.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    DNCON4_CNN.summary()
    return DNCON4_CNN

def DNCON4_with_paras1(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    #RCNN
    # for fsz in filter_sizes:
    #     DNCON4_1D_conv = DNCON4_1D_input
    #     conv_l = Conv1D(filters=filters, kernel_size=fsz, padding='same', activation='relu')
    #     DNCON4_1D_conv = conv_l(DNCON4_1D_conv)
    #     for n in range(nb_layers):
    #         DNCON4_1D_conv = RCL_block(filters, DNCON4_1D_conv,fsz)
        
    #     DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    # if len(filter_sizes)>1:
    #     DNCON4_1D_out = Average(DNCON4_1D_convs)
    # else:
    #     DNCON4_1D_out = DNCON4_1D_convs[0]  
    # DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filters*3),batch_size=batch_size)(DNCON4_1D_out)
    
    #RESATT
    # for fsz in filter_sizes:
    #     DNCON4_1D_conv = DNCON4_1D_input
    #     DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
    #     for i in range(0, 2):
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
    #         DNCON4_1D_conv = Dropout(0.15)(DNCON4_1D_conv)

    #     DNCON4_1D_conv = _attention_layer(filters)(DNCON4_1D_conv)

    #     DNCON4_1D_convs.append(DNCON4_1D_conv)

    # if len(filter_sizes) > 1:
    #     DNCON4_1D_out = Average(DNCON4_1D_convs)
    # else:
    #     DNCON4_1D_out = DNCON4_1D_convs[0]

    # DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filters*3),batch_size=batch_size)(DNCON4_1D_out)
    

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
            DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        
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

def DNCON4_RCIncep_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):

    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

         ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_0 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filters+32, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filters,fsz)
            DNCON4_1D_conv = Dropout(0.3)(DNCON4_1D_conv)#0.3
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)       
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

def RCL_block_2D(input_num_filters,l,nb_row, nb_col, initializer = "he_normal"):
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
def DeepCovRCNN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []

    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        conv_l = Conv1D(filters=filters, kernel_size=fsz, padding='same', activation='relu')
        DNCON4_1D_conv = conv_l(DNCON4_1D_conv)
        
        for n in range(nb_layers):
            DNCON4_1D_conv = RCL_block(filters, DNCON4_1D_conv,fsz)
        
        DNCON4_1D_convs.append(DNCON4_1D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]  

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filters*3),batch_size=batch_size)(DNCON4_1D_out)
    
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
        conv_l = Conv2D(filters=filters, kernel_size=(fsz, fsz), padding='same', activation='relu')
        DNCON4_2D_conv = conv_l(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = RCL_block_2D(filters, DNCON4_2D_conv, fsz, fsz)
        
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

def DeepCovRCNN_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "he_normal", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0):
    
    filter_sizes=win_array
        # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv = Dense(64)(DNCON4_2D_conv)
        DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64)(DNCON4_2D_conv)
        conv_l = Conv2D(filters=filters, kernel_size=(fsz, fsz), padding='same', activation='relu')
        DNCON4_2D_conv = conv_l(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = RCL_block_2D(filters, DNCON4_2D_conv, fsz, fsz)
        
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)
    
    if len(filter_sizes)>1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]  
    ### training on 2D feature 
    
    # DNCON4_flatten = Flatten()(DNCON4_2D_out)
    if loss_function == 'weighted_crossentropy':
        loss = _weighted_binary_crossentropy(weight_p, weight_n)
    else:
        loss = loss_function
    DNCON4_RCNN = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    DNCON4_RCNN.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RCNN.summary()
    return DNCON4_RCNN

def identity_Block_sallow(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters, kernel_size=3, subsample=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    if mode=='sum':
        x = add([x, input])
    elif mode=='concat':
        x = concatenate([x, input], axis=-1)
    return x

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

def identity_Block_sallow_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer='he_normal', dilation_rate=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    # x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    if with_conv_shortcut:
        shortcut = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same")(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        x = Activation("relu")(x)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        x = Activation("relu")(x)
        return x

def identity_Block_deep_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer='he_normal', dilation_rate=(1,1 )):
    # x = Conv2D(filters=np.int32(filters/4), kernel_size=(1, 1), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    # x = Activation("relu")(x)
    # x = Conv2D(filters=np.int32(filters/4), kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    # x = Activation("relu")(x)
    # x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    x = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(input)
    x = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    x = _conv_bn_relu2D(filters=filters, nb_row=1, nb_col = 1, subsample=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    if with_conv_shortcut:
        # shortcut = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), kernel_initializer=kernel_initializer)(input)
        shortcut = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        x = Activation("relu")(x)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        x = Activation("relu")(x)
        return x

def DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=7, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters*2, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters*4, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filters*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)     
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
        DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=7, nb_col=7, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a1 = _conv_relu2D(filters=filters, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b1 = add([DNCON4_2D_conv_a1, DNCON4_2D_conv])

        DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a2 = _conv_relu2D(filters=filters*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b2 = _conv_relu2D(filters=filters*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)

        DNCON4_2D_conv_c2 = add([DNCON4_2D_conv_a2, DNCON4_2D_conv_b2, DNCON4_2D_conv])
        
        DNCON4_2D_conv = _conv_relu2D(filters=filters*4, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_c2)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        
        DNCON4_2D_conv_a3 = _conv_relu2D(filters=filters*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b3 = _conv_relu2D(filters=filters*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv_c3 = _conv_relu2D(filters=filters*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_c2)

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

def DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "he_normal", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0):
    filter_sizes=win_array
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv_in = DNCON4_2D_input
        # DNCON4_2D_conv_bn = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv_in = Dense(64)(DNCON4_2D_conv_in)
        DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64, nb_features=4)(DNCON4_2D_conv_in)
        DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv)
        for idx in range(nb_layers):
            DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a1 = _conv_relu2D(filters=filters, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b1 = add([DNCON4_2D_conv_a1, DNCON4_2D_conv])
        
        # DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz,nb_col=fsz,with_conv_shortcut=True,use_bias=True, mode='concat', kernel_initializer=initializer, dilation_rate=(4,4))
        DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv)
        for idx in range(nb_layers+2):
            DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filters*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a2 = _conv_relu2D(filters=filters*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b2 = _conv_relu2D(filters=filters*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_b1)
        DNCON4_2D_conv = add([DNCON4_2D_conv_a2, DNCON4_2D_conv_b2, DNCON4_2D_conv])
        
        # DNCON4_2D_conv = _conv_relu2D(filters=filters*4, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv)
        # for idx in range(nb_layers):
        #     DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filters*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer)
        # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        # DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
    
        # DNCON4_2D_conv_a3 = _conv_relu2D(filters=filters*3, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        # DNCON4_2D_conv_b3 = _conv_relu2D(filters=filters*3, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)
        # DNCON4_2D_conv_c3 = _conv_relu2D(filters=filters*3, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_c2)
        # DNCON4_2D_conv_a3 = Dense(filters*3)(DNCON4_2D_conv_in)
        # DNCON4_2D_conv_b3 = Dense(filters*3)(DNCON4_2D_conv_b1)
        # DNCON4_2D_conv_c3 = Dense(filters*3)(DNCON4_2D_conv_c2)

        # DNCON4_2D_conv = add([DNCON4_2D_conv_a3, DNCON4_2D_conv_b3, DNCON4_2D_conv_c3, DNCON4_2D_conv])
        # DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=(fsz, fsz), strides=(1, 1),use_bias=use_bias, dilation_rate=(4, 4), kernel_initializer=initializer, padding="same")(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(2, 2))(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(5, 5))(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, subsample=(1, 1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, subsample=(1, 1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    if loss_function == 'weighted_crossentropy':
        loss = _weighted_binary_crossentropy(weight_p, weight_n)
    else:
        loss = loss_function
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    # categorical_crossentropy
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES

def block_inception_a(inputs,filters,kernel_size,use_bias=True):
    inputs_feanum = inputs.shape.as_list()[2]

    branch_0 = _conv_bn_relu1D(filters=filters, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_1 = _conv_bn_relu1D(filters=filters+8, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_2 = _conv_bn_relu1D(filters=filters+16, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_3 = _conv_bn_relu1D(filters=filters+24, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    #x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return net

def block_inception_a_2D(inputs,filters, nb_row, nb_col,subsample=(1, 1), kernel_initializer = "he_normal"):
    inputs_feanum = inputs.shape.as_list()[3]

    branch_0 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1), kernel_initializer = kernel_initializer)(inputs)

    branch_1 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1), kernel_initializer = kernel_initializer)(inputs)
    branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), kernel_initializer = kernel_initializer)(branch_1)

    branch_2 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1), kernel_initializer = kernel_initializer)(inputs)
    branch_2 = _conv_bn_relu2D(filters=filters+8, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), kernel_initializer = kernel_initializer)(branch_2)
    branch_2 = _conv_bn_relu2D(filters=filters+16, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), kernel_initializer = kernel_initializer)(branch_2)

    net = concatenate([branch_0, branch_1, branch_2], axis=-1)
    merge_branch = _conv_bn_relu2D(filters=inputs_feanum, nb_row = 1, nb_col = 1, subsample=(1, 1), kernel_initializer = kernel_initializer)(net)
    outputs = add([inputs, merge_branch])
    return outputs

def DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

         ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_0 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filters+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filters+32, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filters,fsz)
            DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)       
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
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
        shape1 = DNCON4_2D_conv1.shape.as_list()[3]
        ## start inception 1
        branch_0 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        branch_1 = _conv_bn_relu2D(filters=filters+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv2 = Dropout(0.2)(DNCON4_2D_conv2)#0.3
        DNCON4_2D_conv2 = _conv_bn_relu2D(filters=shape1, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv2)
        DNCON4_2D_conv2 = add([DNCON4_2D_conv1, DNCON4_2D_conv2])

        shape2 = DNCON4_2D_conv2.shape.as_list()[3]
        ## start inception 2
        branch_0 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_0 = _conv_bn_relu2D(filters=filters+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_0)

        branch_1 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_1 = _conv_bn_relu2D(filters=filters+16, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)
        branch_1 = _conv_bn_relu2D(filters=filters+32, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)

        DNCON4_2D_conv3 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv3 = Dropout(0.2)(DNCON4_2D_conv3)#0.3
        DNCON4_2D_conv3 = _conv_bn_relu2D(filters=shape2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv)
        DNCON4_2D_conv = add([DNCON4_2D_conv2, DNCON4_2D_conv3])
        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_2D_conv = block_inception_a_2D(DNCON4_2D_conv,filters,nb_row=fsz, nb_col=fsz)
            DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)#0.3
            
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    DNCON4_INCEP = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    # Multi_DNCON4_CNN = multi_gpu_model(DNCON4_INCEP, gpus=2)
    DNCON4_INCEP.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    # DNCON4_INCEP.summary()
    return DNCON4_INCEP

def DeepInception_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "glorot_uniform", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0):

    # load 2D contact features
    filter_sizes=win_array
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)   
    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv = Dense(64)(DNCON4_2D_conv)
        DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64, padding='same')(DNCON4_2D_conv)
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv)
        shape1 = DNCON4_2D_conv1.shape.as_list()[3]
        ## start inception 1
        branch_0 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv1)
        branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv2 = Dropout(0.2)(DNCON4_2D_conv2)#0.3

        shape2 = DNCON4_2D_conv2.shape.as_list()[3]
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=shape2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = add([DNCON4_2D_conv1, DNCON4_2D_conv2])

        ## start inception 2
        branch_0 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv2)
        branch_0 = _conv_bn_relu2D(filters=filters+8, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(branch_0)

        branch_1 = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv2)
        branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(branch_1)
        branch_1 = _conv_bn_relu2D(filters=filters+16, nb_row=fsz, nb_col=fsz, subsample=(1,1), kernel_initializer = initializer)(branch_1)

        DNCON4_2D_conv3 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv3)#0.3

        shape3 = DNCON4_2D_conv.shape.as_list()[3]
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=shape3, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = _conv_bn_relu2D(filters=shape3, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer = initializer)(DNCON4_2D_conv2)
        DNCON4_2D_conv = add([DNCON4_2D_conv1, DNCON4_2D_conv2, DNCON4_2D_conv3])
        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_2D_conv = block_inception_a_2D(DNCON4_2D_conv,filters,nb_row=fsz, nb_col=fsz, kernel_initializer = initializer)
            DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)#0.3

        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)    
        # DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=use_bias, dilation_rate=(2, 2), kernel_initializer=initializer, padding="same")(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1), kernel_initializer = initializer)(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    print("out shape", DNCON4_2D_out.shape.as_list())
    # DNCON4_flatten = Flatten()(DNCON4_2D_out)
    DNCON4_INCEP = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    if loss_function == 'weighted_crossentropy':
        loss = _weighted_binary_crossentropy(weight_p, weight_n)
    else:
        loss = loss_function
    # categorical_crossentropy  # binary_crossentropy 
    DNCON4_INCEP.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_INCEP.summary()
    return DNCON4_INCEP

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
def DeepCovResAtt_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        for i in range(0, nb_layers):
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = identity_Block(DNCON4_1D_conv, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            DNCON4_1D_conv = Dropout(0.15)(DNCON4_1D_conv)

        DNCON4_1D_conv = _attention_layer(filters)(DNCON4_1D_conv)

        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,filters*3),batch_size=batch_size)(DNCON4_1D_out)
    
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
        for i in range(0, nb_layers):
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = identity_Block_2D(DNCON4_2D_conv, filters=filters, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            DNCON4_2D_conv = Dropout(0.15)(DNCON4_2D_conv)

        DNCON4_2D_conv = _attention_layer(filters)(DNCON4_2D_conv)
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

def DeepCRMN_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        for i in range(0, nb_layers):
            cnn = _conv_bn_relu1D(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
            res = identity_Block_CRMN(cnn, filters=filters, kernel_size=fsz, with_conv_shortcut=False, use_bias=True)
            cnnres = add([cnn, res])
            cnnres = Dropout(0.2)(cnnres)

            lstm = LSTM(filters, return_sequences=True)(cnnres)
            Lstmlayer = LSTM(filters, return_sequences=True)(lstm)
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
            cnn = _conv_bn_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
            res = identity_Block_CRMN_2D(cnn, filters=filters, nb_row=fsz, nb_col=fsz, with_conv_shortcut=False)
            cnnres = add([cnn, res])
            cnnres = Dropout(0.2)(cnnres)
            print(cnnres.shape)
            trans2lstm = Reshape((5, cnnres.shape.as_list()[1], cnnres.shape.as_list()[1], -1))(cnnres)
            lstm = ConvLSTM2D(filters=filters, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(trans2lstm)
            Lstmlayer = ConvLSTM2D(filters=filters, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(lstm)
            DNCON4_2D_conv = cnnres

        LastLstm = ConvLSTM2D(filters=filters, kernel_size=(fsz, fsz), padding='same', dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(Lstmlayer)
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

def DeepFracNet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filters,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        for i in range(0, nb_layers):
            DNCON4_1D_conv = fractal_block(filters=filters, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
            DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
            DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_relu1D(filters=filters, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
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
            DNCON4_2D_conv = fractal_block_2D(filters=filters, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
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


