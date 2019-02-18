import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers.convolutional import Conv2D
from keras.layers import Activation
from keras import backend as K
from keras.activations import tanh, softmax,relu
from keras.layers.normalization import BatchNormalization

"""

Maxout OP from https://arxiv.org/abs/1302.4389

Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.

Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


class MaxoutConv2D_Test(Layer):
    """
    Convolution Layer followed by Maxout activation as described 
    in https://arxiv.org/abs/1505.03540.
    
    Parameters
    ----------
    
    kernel_size: kernel_size parameter for Conv2D
    output_dim: final number of filters after Maxout
    nb_features: number of filter maps to take the Maxout over; default=4
    padding: 'same' or 'valid'
    first_layer: True if x is the input_tensor
    input_shape: Required if first_layer=True
    
    """
    # linear
    # elu
    # softplus
    # softsign
    # relu
    # tanh
    # sigmoid
    # hard_sigmoid
    # tanh
    
    def __init__(self, kernel_size, output_dim, filters=4, padding='same', activation = "relu", **kwargs):
        
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.filters = filters
        self.padding = padding
        self.use_bias = False
        self.kernel_initializer = "glorot_uniform"
        self.bias_initializer = "zeros"

        super(MaxoutConv2D_Test, self).__init__(**kwargs)

    def call(self, x):

        output = None
        print("before",len(x.shape))
        # x = K.reshape(x, (x.shape)) 
        # print("after",len(x.shape))
        for _ in range(self.output_dim):

            # conv = K.conv2d(x, self.weight, padding='same',data_format='channels_last')
            # conv = K.squeeze(conv, 2)  # remove the dummy dimension

            # conv = Conv2D(self.filters, self.kernel_size, padding=self.padding)(x)
            inputs = x
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=(1,1), 
                padding=self.padding,
                data_format='channels_last',
                dilation_rate=(1,1))

            activa = Activation("relu")(outputs)
            # activa = BatchNormalization(axis=-1)(activa)
            maxout_out = K.max(activa, axis=-1, keepdims=True)

            if output is not None:
                output = K.concatenate([output, maxout_out], axis=-1)

            else:
                output = maxout_out

        return output

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)
        else:
            self.bias = None
        # # self.W = K.variable(an_init_numpy_array)
        super(MaxoutConv2D_Test, self).build(input_shape)

    def get_config(self):
        config = {'name':self.__class__.__name__,'kernel_size':self.kernel_size, 'output_dim': self.output_dim}
        base_config = super(MaxoutConv2D_Test, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        # return{'name':self.__class__.__name__,'kernel_size':self.kernel_size, 'output_dim': self.output_dim}

    def compute_output_shape(self, input_shape):
        input_height= input_shape[1]
        input_width = input_shape[2]
        

        output_height = input_height
        output_width = input_width
        
        return (input_shape[0], output_height, output_width, self.output_dim)

