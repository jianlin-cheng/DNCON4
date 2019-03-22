
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


def _in_relu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        act = _in_relu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(act)
        return conv
    return f

def _in_relu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("relu")(norm)

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

def _conv_in_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

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



#get binary y lable
def getY(true_file, min_seq_sep, l_max):
  # calcualte the length of the protein (the first feature)
  L = 0
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      L = line.strip().split()
      L = len(L)
      break
  Y = np.zeros((l_max, l_max))
  i = 0
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      this_line = line.strip().split()
      if len(this_line) != L:
        print("\nThis_line = %i, L = %i, Lable file %s error!\n"%(len(this_line), L, true_file))
        Y = [0]
        return  Y
      Y[i, 0:L] = np.asarray(this_line)
      i = i + 1
  for p in range(0,L):
    for q in range(0,L):
      # updated only for the last project 'p19' to test the effect
      if ( abs(q - p) < min_seq_sep):
        Y[p][q] = 0
  Y = Y.flatten()
  return Y


contact_feature_num_2D=4
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

# Last activation
block = _in_relu(block)
DNCON4_2D_conv = block
loss_function = 'categorical_crossentropy'
DNCON4_2D_conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1),use_bias=True, padding="same", kernel_regularizer=None)(DNCON4_2D_conv)
# norm = BatchNormalization(axis=-1)(DNCON4_2D_conv)
DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
DNCON4_2D_conv = Dense(42, activation='softmax')(DNCON4_2D_conv) 
#loss = _unweighted_categorical_crossentropy()

DNCON4_2D_out = DNCON4_2D_conv
DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
# categorical_crossentropy
DNCON4_RES.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='nadam')
DNCON4_RES.summary()


path_of_lists='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
path_of_Y='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/dist_map/'
pdb_name='3CRY-A'
targetfile = path_of_Y + pdb_name + '.txt'
l_max=188
Y1 = getY(targetfile, 0, l_max)
Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
Y= (np.arange(Y1.max()+1) == Y1[...,None]).astype(int) # L*L*1*42
Y = Y.reshape(1,Y.shape[0],Y.shape[1],Y.shape[3])

inputdata = np.random.rand(1,l_max,l_max,4)

DNCON4_RES.fit(inputdata,Y)

prediction = DNCON4_RES.predict(inputdata)# 1*L*L:42


check = prediction[0,0,0,:] 
np.sum(check)


## get probability for  <=8
pred_8A= prediction[:,:,:,0:8].sum(axis=-1)


#### check the dimension 

path_lists = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'



path_of_lists='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/lists-test-train/'
path_of_Y='/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/dist_map/'
pdb_name='1RB9-A'
targetfile = path_of_Y + pdb_name + '.txt'
l_max=52
Y1 = getY(targetfile, 0, l_max)
Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
Y= (np.arange(42) == Y1[...,None]).astype(int) # L*L*1*42
Y = Y.reshape(1,Y.shape[0],Y.shape[1],Y.shape[3])

