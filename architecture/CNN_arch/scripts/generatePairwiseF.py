class generatePairwiseF(Layer):
    '''
        (l,n) -> (l*l,3n)
    '''
    def __init__(self, output_shape, batch_size, **kwargs):
        super(generatePairwiseF, self).__init__(**kwargs)
        self._output_shape = tuple(output_shape)
        self._batch_size = batch_size
        super(generatePairwiseF, self).__init__()
    
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