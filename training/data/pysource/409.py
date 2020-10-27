

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import types as python_types

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import activations
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.engine import InputSpec
from tensorflow.python.keras._impl.keras.engine import Layer
from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras._impl.keras.utils.generic_utils import func_dump
from tensorflow.python.keras._impl.keras.utils.generic_utils import func_load
from tensorflow.python.keras._impl.keras.utils.generic_utils import has_arg
from tensorflow.python.layers import core as tf_core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export

@tf_export('keras.layers.Masking')
class Masking(Layer):

  def __init__(self, mask_value=0., **kwargs):
    super(Masking, self).__init__(**kwargs)
    self.supports_masking = True
    self.mask_value = mask_value

  def compute_mask(self, inputs, mask=None):
    return K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    boolean_mask = K.any(
        math_ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    return inputs * math_ops.cast(boolean_mask, inputs.dtype)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.Dropout')
class Dropout(tf_core_layers.Dropout, Layer):

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(rate=rate,
                                  noise_shape=noise_shape,
                                  seed=seed,
                                  **kwargs)
    self.supports_masking = True

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    output = super(Dropout, self).call(inputs, training=training)
    if not context.executing_eagerly() and training is K.learning_phase():
      output._uses_learning_phase = True  
    return output

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.SpatialDropout1D')
class SpatialDropout1D(Dropout):

  def __init__(self, rate, **kwargs):
    super(SpatialDropout1D, self).__init__(rate, **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    noise_shape = (input_shape[0], 1, input_shape[2])
    return noise_shape

@tf_export('keras.layers.SpatialDropout2D')
class SpatialDropout2D(Dropout):

  def __init__(self, rate, data_format=None, **kwargs):
    super(SpatialDropout2D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=4)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, input_shape[3])

@tf_export('keras.layers.SpatialDropout3D')
class SpatialDropout3D(Dropout):

  def __init__(self, rate, data_format=None, **kwargs):
    super(SpatialDropout3D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=5)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])

@tf_export('keras.layers.Activation')
class Activation(Layer):

  def __init__(self, activation, **kwargs):
    super(Activation, self).__init__(**kwargs)
    self.supports_masking = True
    self.activation = activations.get(activation)

  def call(self, inputs):
    return self.activation(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'activation': activations.serialize(self.activation)}
    base_config = super(Activation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.Reshape')
class Reshape(Layer):

  def __init__(self, target_shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    output_shape = list(output_shape)
    msg = 'total size of new array must be unchanged'

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
      if dim < 0:
        if unknown is None:
          unknown = index
        else:
          raise ValueError('Can only specify one unknown dimension.')
      else:
        known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
      if known == 0 or original % known != 0:
        raise ValueError(msg)
      output_shape[unknown] = original // known
    elif original != known:
      raise ValueError(msg)
    return output_shape

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if None in input_shape[1:]:
      output_shape = [input_shape[0]]
      output_shape += tuple(s if s != -1 else None for s in self.target_shape)
    else:
      output_shape = [input_shape[0]]
      output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                  self.target_shape)
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    return array_ops.reshape(inputs,
                             (array_ops.shape(inputs)[0],) + self.target_shape)

  def get_config(self):
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.Permute')
class Permute(Layer):

  def __init__(self, dims, **kwargs):
    super(Permute, self).__init__(**kwargs)
    self.dims = tuple(dims)
    self.input_spec = InputSpec(ndim=len(self.dims) + 1)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = copy.copy(input_shape)
    for i, dim in enumerate(self.dims):
      target_dim = input_shape[dim]
      output_shape[i + 1] = target_dim
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    return array_ops.transpose(inputs, perm=(0,) + self.dims)

  def get_config(self):
    config = {'dims': self.dims}
    base_config = super(Permute, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.Flatten')
class Flatten(tf_core_layers.Flatten, Layer):
  pass

@tf_export('keras.layers.RepeatVector')
class RepeatVector(Layer):

  def __init__(self, n, **kwargs):
    super(RepeatVector, self).__init__(**kwargs)
    self.n = n
    self.input_spec = InputSpec(ndim=2)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

  def call(self, inputs):
    return K.repeat(inputs, self.n)

  def get_config(self):
    config = {'n': self.n}
    base_config = super(RepeatVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.Lambda')
class Lambda(Layer):

  def __init__(self, function, output_shape=None, mask=None, arguments=None,
               **kwargs):
    super(Lambda, self).__init__(**kwargs)
    self.function = function
    self.arguments = arguments if arguments else {}
    if mask is not None:
      self.supports_masking = True
    self.mask = mask
    if output_shape is None:
      self._output_shape = None
    elif isinstance(output_shape, (tuple, list)):
      self._output_shape = tuple(output_shape)
    else:
      if not callable(output_shape):
        raise TypeError('In Lambda, `output_shape` '
                        'must be a list, a tuple, or a function.')
      self._output_shape = output_shape

  def _compute_output_shape(self, input_shape):
    input_shape = tuple(tensor_shape.TensorShape(input_shape).as_list())

    if self._output_shape is None:
      x = K.placeholder(shape=input_shape)
      x = self.call(x)
      if isinstance(x, list):
        return [tensor_shape.TensorShape(K.int_shape(x_elem)) for x_elem in x]
      else:
        return tensor_shape.TensorShape(K.int_shape(x))
    elif isinstance(self._output_shape, (tuple, list)):
      if isinstance(input_shape, list):
        num_samples = input_shape[0][0]
      else:
        num_samples = input_shape[0] if input_shape else None
      return tensor_shape.TensorShape((num_samples,) +
                                      tuple(self._output_shape))
    else:
      shape = self._output_shape(input_shape)
      if not isinstance(shape, (list, tuple)):
        raise ValueError(
            '`output_shape` function must return a tuple or a list of tuples.')
      if isinstance(shape, list):
        if isinstance(shape[0], int) or shape[0] is None:
          shape = tuple(shape)
      return tensor_shape.TensorShape(shape)

  def call(self, inputs, mask=None):
    arguments = self.arguments
    if has_arg(self.function, 'mask'):
      arguments['mask'] = mask
    return self.function(inputs, **arguments)

  def compute_mask(self, inputs, mask=None):
    if callable(self.mask):
      return self.mask(inputs, mask)
    return self.mask

  def get_config(self):
    if isinstance(self.function, python_types.LambdaType):
      function = func_dump(self.function)
      function_type = 'lambda'
    else:
      function = self.function.__name__
      function_type = 'function'

    if isinstance(self._output_shape, python_types.LambdaType):
      output_shape = func_dump(self._output_shape)
      output_shape_type = 'lambda'
    elif callable(self._output_shape):
      output_shape = self._output_shape.__name__
      output_shape_type = 'function'
    else:
      output_shape = self._output_shape
      output_shape_type = 'raw'

    config = {
        'function': function,
        'function_type': function_type,
        'output_shape': output_shape,
        'output_shape_type': output_shape_type,
        'arguments': self.arguments
    }
    base_config = super(Lambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    globs = globals()
    if custom_objects:
      globs = dict(list(globs.items()) + list(custom_objects.items()))
    function_type = config.pop('function_type')
    if function_type == 'function':
      function = deserialize_keras_object(
          config['function'],
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
      function = func_load(config['function'], globs=globs)
    else:
      raise TypeError('Unknown function type:', function_type)

    output_shape_type = config.pop('output_shape_type')
    if output_shape_type == 'function':
      output_shape = deserialize_keras_object(
          config['output_shape'],
          custom_objects=custom_objects,
          printable_module_name='output_shape function in Lambda layer')
    elif output_shape_type == 'lambda':
      output_shape = func_load(config['output_shape'], globs=globs)
    else:
      output_shape = config['output_shape']

    if 'arguments' in config:
      for key in config['arguments']:
        if isinstance(config['arguments'][key], dict):
          arg_dict = config['arguments'][key]
          if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
            config['arguments'][key] = np.array(arg_dict['value'])

    config['function'] = function
    config['output_shape'] = output_shape
    return cls(**config)

@tf_export('keras.layers.Dense')
class Dense(tf_core_layers.Dense, Layer):

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(Dense, self).__init__(
        units,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
    self.supports_masking = True

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.layers.ActivityRegularization')
class ActivityRegularization(Layer):

  def __init__(self, l1=0., l2=0., **kwargs):
    super(ActivityRegularization, self).__init__(
        activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs)
    self.supports_masking = True
    self.l1 = l1
    self.l2 = l2

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'l1': self.l1, 'l2': self.l2}
    base_config = super(ActivityRegularization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
EOF
