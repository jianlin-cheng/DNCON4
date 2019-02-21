
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Lambda, add, concatenate,ConvLSTM2D
from keras.layers.convolutional import Conv2D,Conv1D
from keras.optimizers import Adam
from keras.activations import tanh, softmax

import numpy as np
import tensorflow as tf
import keras.backend as K
def _weights_mean_squared_error(y_ture, y_pred):
    y_mean = K.mean(y_ture)
    weights = 1/(1 + K.square(y_pred/y_mean))
    return K.mean(K.square((y_pred-y_ture)*weights))


def weighted_categorical_crossentropy(weights):
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