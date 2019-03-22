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
from keras.layers import Input, Dense, Reshape, Activation, Flatten, Embedding, merge, Dropout, Lambda, add, concatenate,ConvLSTM2D, LSTM, Average, MaxPooling2D, multiply
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.generic_utils import Progbar
from keras.constraints import maxnorm
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization
from keras.activations import tanh, softmax
from keras import metrics, initializers, utils, regularizers
import numpy as np

import tensorflow as tf
from maxout import MaxoutConv2D, max_out
from maxout_test import MaxoutConv2D_Test
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D_test(filters, kernel_size, strides,use_bias=True):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer="he_normal", activation='relu', padding="same")(input)
        return Activation("sigmoid")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu1D(filters, kernel_size, strides,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_relu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _bn_relu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        act = _bn_relu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(act)
        return conv
    return f

def _in_relu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        act = _in_relu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(act)
        return conv
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_bn_relu2D_sigmoid(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("sigmoid")(norm)
    return f

def _conv_bn_relu2D_softmax(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("softmax")(norm)
    return f

def _conv_in_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_relu1D(filters, kernel_size, strides, use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same")(input)
        return Activation("relu")(conv)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_relu2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides, use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        # norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(conv)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_softmax1D(filters, kernel_size, strides, name,use_bias=True, kernel_initializer = "he_normal"):
    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same",name="%s_conv" % name)(input)
        norm = BatchNormalization(axis=-1,name="%s_nor" % name)(conv)
        return Dense(units=3, kernel_initializer=kernel_initializer,name="%s_softmax" % name, activation="softmax")(norm)
    
    return f

# Helper to build a conv -> BN -> softmax block
def _conv_bn_sigmoid2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
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


def _weighted_mean_squared_error(weight):

    def loss(y_true, y_pred):
        #set 20A as thresold
        # y_bool = Lambda(lambda x: x <= 20.0)(y_pred)
        y_bool = K.cast((y_true <= 20.0), dtype='float32')
        y_bool_invert = K.cast((y_true > 20.0), dtype='float32')
        y_mean = K.mean(y_true)
        y_pred_below = y_pred * y_bool 
        y_pred_upper = y_pred * y_bool_invert 
        y_true_below = y_true * y_bool 
        y_true_upper = y_true * y_bool_invert 
        # y_pred_upper = multiply([y_pred, y_bool_invert])
        # y_true_below = multiply([y_true, y_bool])
        # y_true_upper = multiply([y_true, y_bool_invert])
        weights1 = 1
        # weights2 = 0
        weights2 = 1/(1 + K.square(y_pred_upper/y_mean))
        return K.mean(K.square((y_pred_below-y_true_below))*weights1) + K.mean(K.square((y_pred_upper-y_true_upper))*weights2)
        # return add([K.mean(K.square((y_pred_below-y_true_below))*weights1), K.mean(K.square((y_pred_upper-y_true_upper))*weights2)], axis= -1)
    return loss

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

def _unweighted_categorical_crossentropy():
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    #weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred)
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

def MaxoutAct(input, filters, kernel_size, output_dim, padding='same', activation = "relu"):
    output = None
    for _ in range(output_dim):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        activa = Activation(activation)(conv)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(activa)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

def MaxoutCov(input, output_dim):
    output = None
    for i in range(output_dim):
        section = Lambda(lambda x:x[:,:,:,2*i:2*i+1])(input)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(section)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

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
        DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, padding='same', activation = "relu")

        # DNCON4_2D_conv = Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same')(DNCON4_2D_conv)
        # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        # DNCON4_2D_conv = MaxoutCov(DNCON4_2D_conv, output_dim=64)

        # all param 645473
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, strides=(1, 1))(DNCON4_2D_conv)
            # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
            DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = Dropout(0.4)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters*2, nb_row=fsz, nb_col=fsz, strides=(1, 1))(DNCON4_2D_conv)
            # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
            DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)
        for i in range(0,nb_layers):
            DNCON4_2D_conv = _conv_relu2D(filters=filters*4, nb_row=fsz, nb_col=fsz, strides=(1, 1))(DNCON4_2D_conv)
            # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
            DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        
        #all param 1025985
        # for i in range(0,nb_layers):
        #     DNCON4_2D_conv = _conv_relu2D(filters=filters, nb_row=fsz, nb_col=fsz, strides=(1, 1))(DNCON4_2D_conv)
        #     DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        #     # DNCON4_2D_conv = Dropout(0.1)(DNCON4_2D_conv)

        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1))(DNCON4_2D_conv)
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

def identity_Block_sallow(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters, kernel_size=3, strides=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(x)
    if mode=='sum':
        x = add([x, input])
    elif mode=='concat':
        x = concatenate([x, input], axis=-1)
    return x

def identity_Block_deep(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters*2, kernel_size=1, strides=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(x)
    x = _conv_relu1D(filters=filters, kernel_size=1, strides=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, strides=1,use_bias=use_bias)(input)
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
    x = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    x = Activation("relu")(x)
    x = InstanceNormalization(axis=-1)(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
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
    x = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(input)
    x = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    x = _conv_bn_relu2D(filters=filters, nb_row=1, nb_col = 1, strides=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer)(x)
    if with_conv_shortcut:
        # shortcut = _conv_bn_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, strides=(1, 1), kernel_initializer=kernel_initializer)(input)
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

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    # stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    # stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    stride_width = 1
    stride_height = 1
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    #     shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
    #                       kernel_size=(1, 1),
    #                       strides=(stride_width, stride_height),
    #                       padding="valid",
    #                       kernel_initializer="he_normal",
    #                       kernel_regularizer=regularizers.l2(0.0001))(input)
    if not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal")(input)
    return add([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                # init_strides = (2, 2)
                init_strides = (1, 1)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal")(input)
                           # ,
                           # kernel_regularizer=regularizers.l2(1e-4)
        else:
            conv1 = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3,
                                  strides=init_strides)(input)

        residual = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3)(conv1)
        return _shortcut(input, residual)
    return f

def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _in_relu_conv2D(filters=filters, nb_row=1, nb_col=1,
                                     strides=init_strides)(input)

        conv_3_3 = _in_relu_conv2D(filters=filters, nb_row=3, nb_col=3)(conv_1_1)
        residual = _in_relu_conv2D(filters=filters * 4, nb_row=1, nb_col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f

def DeepResnet_with_paras_2D(kernel_size,feature_2D_num,use_bias,hidden_type,filters,nb_layers,opt, initializer = "he_normal", loss_function = "weighted_BCE", weight_p=1.0, weight_n=1.0):

    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    _handle_dim_ordering()
    ######################### now merge new data to new architecture
    # regularizers.l2(0.01)
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []

    DNCON4_2D_conv_in = DNCON4_2D_input
    # DNCON4_2D_conv_bn = BatchNormalization(axis=-1)(DNCON4_2D_conv)
    DNCON4_2D_conv_in = Dense(64)(DNCON4_2D_conv_in)
    # # DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64, nb_features=4)(DNCON4_2D_conv_in) 
    DNCON4_2D_conv = MaxoutAct(DNCON4_2D_conv_in, filters=4, kernel_size=(1,1), output_dim=64, padding='same', activation = "relu")

    # ######This is original residual
    DNCON4_2D_conv = _conv_in_relu2D(filters=64, nb_row=7, nb_col=7, strides=(1, 1))(DNCON4_2D_conv)
    block = DNCON4_2D_conv
    filters = 64
    # repetitions = [2, 2, 2, 2]
    repetitions = [3, 4, 6, 3]
    for i, r in enumerate(repetitions):
        block = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        # block = _residual_block(bottleneck, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        # filters *= 2
    # Last activation
    block = _in_relu(block)
    DNCON4_2D_conv = block

    if loss_function == 'weighted_BCE':
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        loss = _weighted_binary_crossentropy(weight_p, weight_n) 
    elif loss_function == 'unweighted_BCE':
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        loss = 'binary_crossentropy' 
    elif loss_function == 'weighted_CCE':
        loss_function = _weighted_categorical_crossentropy(weight_p)
    elif loss_function == 'weighted_MSE':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited2':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited3':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited4':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited16':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited17':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited19':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited21':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited22':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited23':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'weighted_MSE_limited25':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    elif loss_function == 'sigmoid_MSE':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = 'mean_squared_error'
    elif loss_function == 'MSE_limited':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = 'mean_squared_error'
    elif loss_function == 'MSE_limited2':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = 'mean_squared_error'
    elif loss_function == 'MSE_limited3':
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = 'mean_squared_error'
    elif loss_function == 'categorical_crossentropy':
        #DNCON4_2D_conv = _conv_bn_relu2D_softmax(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                             kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DNCON4_2D_conv)
        # norm = BatchNormalization(axis=-1)(conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = Dense(42, activation='softmax')(DNCON4_2D_conv) 
        #loss = _unweighted_categorical_crossentropy()
        loss= 'categorical_crossentropy'
    elif loss_function == 'weighted_MSElimited20_disterror':
        DNCON4_2D_conv = _conv_in_relu2D(filters=2, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)        
        # DNCON4_2D_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1,1),kernel_initializer=initializer, padding="same")(DNCON4_2D_conv)
        # DNCON4_2D_conv = InstanceNormalization(axis=-1)(DNCON4_2D_conv)
        loss = _weighted_mean_squared_error(weight_p)
    else:
        DNCON4_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DNCON4_2D_conv)
        loss = loss_function
    DNCON4_2D_out = DNCON4_2D_conv
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    # categorical_crossentropy
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES
