import lasagne

RAW_CHANNELS = 441

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import instance_norm

def build_cnn(input_var=None):

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = instance_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(5,5), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = instance_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(5,5), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = instance_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(1, RAW_CHANNELS, None, None), input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=64*2, filter_size=1,
            nonlinearity=None,
            W=lasagne.init.GlorotUniform('relu')), alpha=0.01)

    network = lasagne.layers.FeaturePoolLayer(network, 2)

    # first stack of residual blocks, output is 16 x 32 x 32
    for i in range(18):
        network = residual_block(network)
    # for i in range(18):
        # network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
                # network, num_filters=64, filter_size=5, pad='same',
                # nonlinearity=lasagne.nonlinearities.rectify,
                # W=lasagne.init.GlorotUniform('relu')), gamma=None, alpha=0.01)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=1, filter_size=1,
            nonlinearity=lasagne.nonlinearities.sigmoid), alpha=0.01)

    return network
