from tensorflow.python.keras.engine import Layer


class ILayer(Layer):
    def __init__(self,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ILayer, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = (dilation_rate, dilation_rate) if isinstance(dilation_rate, int) else dilation_rate
        return

