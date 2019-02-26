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