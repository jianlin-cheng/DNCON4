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
    
    def __init__(self, kernel_size, output_dim, nb_features=4, padding='same', activation = "relu", **kwargs):
        
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.nb_features = nb_features
        self.padding = padding
        super(MaxoutConv2D_Test, self).__init__(**kwargs)

    def call(self, x):

        output = None
        for _ in range(self.output_dim):
            
            conv = Conv2D(self.nb_features, self.kernel_size, padding=self.padding)(x)
            # norm = BatchNormalization(axis=-1)(conv)
            activa = Activation("relu")(conv)
            maxout_out = K.max(activa, axis=-1, keepdims=True)

            if output is not None:
                output = K.concatenate([output, maxout_out], axis=-1)

            else:
                output = maxout_out

        return output

    def compute_output_shape(self, input_shape):
        input_height= input_shape[1]
        input_width = input_shape[2]
        

        output_height = input_height
        output_width = input_width
        
        return (input_shape[0], output_height, output_width, self.output_dim)

