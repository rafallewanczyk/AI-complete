

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export

class Pooling1D(Layer):

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(Pooling1D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=3)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      inputs = array_ops.expand_dims(inputs, 1)
      pool_shape = (1, 1) + self.pool_size + (1,)
      strides = (1, 1) + self.strides + (1,)
      data_format = 'NHWC'
    else:
      inputs = array_ops.expand_dims(inputs, 2)
      pool_shape = (1, 1, 1) + self.pool_size
      strides = (1, 1, 1) + self.strides
      data_format = 'NCHW'

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=data_format)

    if self.data_format == 'channels_last':
      return array_ops.squeeze(outputs, 1)
    else:
      return array_ops.squeeze(outputs, 2)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    length = conv_utils.conv_output_length(input_shape[1], self.pool_size[0],
                                           self.padding, self.strides[0])
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding
    }
    base_config = super(Pooling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.MaxPool1D', 'keras.layers.MaxPooling1D')
class MaxPooling1D(Pooling1D):

  def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format=None, **kwargs):

    super(MaxPooling1D, self).__init__(
        nn.max_pool,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)

@tf_export('keras.layers.AveragePooling1D', 'keras.layers.AvgPool1D')
class AveragePooling1D(Pooling1D):

  def __init__(self, pool_size=2, strides=None,
               padding='valid', data_format=None, **kwargs):
    super(AveragePooling1D, self).__init__(
        nn.avg_pool,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        **kwargs)

class Pooling2D(Layer):

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(Pooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.MaxPool2D', 'keras.layers.MaxPooling2D')
class MaxPooling2D(Pooling2D):

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling2D, self).__init__(
        nn.max_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)

@tf_export('keras.layers.AveragePooling2D', 'keras.layers.AvgPool2D')
class AveragePooling2D(Pooling2D):

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(
        nn.avg_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)

class Pooling3D(Layer):

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(Pooling3D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_function = pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def call(self, inputs):
    pool_shape = (1,) + self.pool_size + (1,)
    strides = (1,) + self.strides + (1,)

    if self.data_format == 'channels_first':
      inputs = array_ops.transpose(inputs, (0, 2, 3, 4, 1))

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper())

    if self.data_format == 'channels_first':
      outputs = array_ops.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      len_dim1 = input_shape[2]
      len_dim2 = input_shape[3]
      len_dim3 = input_shape[4]
    else:
      len_dim1 = input_shape[1]
      len_dim2 = input_shape[2]
      len_dim3 = input_shape[3]
    len_dim1 = conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                                             self.padding, self.strides[0])
    len_dim2 = conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                                             self.padding, self.strides[1])
    len_dim3 = conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                                             self.padding, self.strides[2])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], len_dim1, len_dim2, len_dim3])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], len_dim1, len_dim2, len_dim3, input_shape[4]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(Pooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.MaxPool3D', 'keras.layers.MaxPooling3D')
class MaxPooling3D(Pooling3D):

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling3D, self).__init__(
        nn.max_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)

@tf_export('keras.layers.AveragePooling3D', 'keras.layers.AvgPool3D')
class AveragePooling3D(Pooling3D):

  def __init__(self,
               pool_size=(2, 2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling3D, self).__init__(
        nn.avg_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)

class GlobalPooling1D(Layer):

  def __init__(self, **kwargs):
    super(GlobalPooling1D, self).__init__(**kwargs)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

  def call(self, inputs):
    raise NotImplementedError

@tf_export('keras.layers.GlobalAveragePooling1D',
           'keras.layers.GlobalAvgPool1D')
class GlobalAveragePooling1D(GlobalPooling1D):

  def call(self, inputs):
    return backend.mean(inputs, axis=1)

@tf_export('keras.layers.GlobalMaxPool1D', 'keras.layers.GlobalMaxPooling1D')
class GlobalMaxPooling1D(GlobalPooling1D):

  def call(self, inputs):
    return backend.max(inputs, axis=1)

class GlobalPooling2D(Layer):

  def __init__(self, data_format=None, **kwargs):
    super(GlobalPooling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape([input_shape[0], input_shape[3]])
    else:
      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(GlobalPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.GlobalAveragePooling2D',
           'keras.layers.GlobalAvgPool2D')
class GlobalAveragePooling2D(GlobalPooling2D):

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.mean(inputs, axis=[1, 2])
    else:
      return backend.mean(inputs, axis=[2, 3])

@tf_export('keras.layers.GlobalMaxPool2D', 'keras.layers.GlobalMaxPooling2D')
class GlobalMaxPooling2D(GlobalPooling2D):

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.max(inputs, axis=[1, 2])
    else:
      return backend.max(inputs, axis=[2, 3])

class GlobalPooling3D(Layer):

  def __init__(self, data_format=None, **kwargs):
    super(GlobalPooling3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape([input_shape[0], input_shape[4]])
    else:
      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

  def call(self, inputs):
    raise NotImplementedError

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(GlobalPooling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.GlobalAveragePooling3D',
           'keras.layers.GlobalAvgPool3D')
class GlobalAveragePooling3D(GlobalPooling3D):

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.mean(inputs, axis=[1, 2, 3])
    else:
      return backend.mean(inputs, axis=[2, 3, 4])

@tf_export('keras.layers.GlobalMaxPool3D', 'keras.layers.GlobalMaxPooling3D')
class GlobalMaxPooling3D(GlobalPooling3D):

  def call(self, inputs):
    if self.data_format == 'channels_last':
      return backend.max(inputs, axis=[1, 2, 3])
    else:
      return backend.max(inputs, axis=[2, 3, 4])

AvgPool1D = AveragePooling1D
MaxPool1D = MaxPooling1D
AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
AvgPool3D = AveragePooling3D
MaxPool3D = MaxPooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
EOF
