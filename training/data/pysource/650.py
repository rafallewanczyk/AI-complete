

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.Masking')
class Masking(Layer):

  def __init__(self, mask_value=0., **kwargs):
    super(Masking, self).__init__(**kwargs)
    self.supports_masking = True
    self.mask_value = mask_value
    self._compute_output_and_mask_jointly = True

  def compute_mask(self, inputs, mask=None):
    return K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    boolean_mask = K.any(
        math_ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
    outputs._keras_mask = array_ops.squeeze(boolean_mask, axis=-1)  
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.layers.Dropout')
class Dropout(Layer):

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor_v2(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = tf_utils.smart_cond(training,
                                 dropped_inputs,
                                 lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.layers.SpatialDropout1D')
class SpatialDropout1D(Dropout):

  def __init__(self, rate, **kwargs):
    super(SpatialDropout1D, self).__init__(rate, **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    noise_shape = (input_shape[0], 1, input_shape[2])
    return noise_shape

@keras_export('keras.layers.SpatialDropout2D')
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

@keras_export('keras.layers.SpatialDropout3D')
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

@keras_export('keras.layers.Activation')
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

@keras_export('keras.layers.Reshape')
class Reshape(Layer):

  def __init__(self, target_shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    output_shape = list(output_shape)
    msg = ('total size of new array must be unchanged, '
           'input_shape = {}, output_shape = {}'
           .format(input_shape, output_shape))

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
    result = array_ops.reshape(
        inputs, (array_ops.shape(inputs)[0],) + self.target_shape)
    if not context.executing_eagerly():
      result.set_shape(self.compute_output_shape(inputs.shape))
    return result

  def get_config(self):
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.layers.Permute')
class Permute(Layer):

  def __init__(self, dims, **kwargs):
    super(Permute, self).__init__(**kwargs)
    self.dims = tuple(dims)
    if sorted(dims) != list(range(1, len(dims) + 1)):
      raise ValueError(
          'Invalid permutation `dims` for Permute Layer: %s. '
          'The set of indices in `dims` must be consecutive and start from 1.' %
          (dims,))
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

@keras_export('keras.layers.Flatten')
class Flatten(Layer):

  def __init__(self, data_format=None, **kwargs):
    super(Flatten, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(min_ndim=1)
    self._channels_first = self.data_format == 'channels_first'

  def call(self, inputs):
    if self._channels_first:
      rank = inputs.shape.rank
      if rank and rank > 1:
        permutation = [0]
        permutation.extend(range(2, rank))
        permutation.append(1)
        inputs = array_ops.transpose(inputs, perm=permutation)

    if context.executing_eagerly():
      flattened_shape = constant_op.constant([inputs.shape[0], -1])
      return gen_array_ops.reshape(inputs, flattened_shape)
    else:
      input_shape = inputs.shape
      rank = input_shape.rank
      if rank == 1:
        return array_ops.expand_dims_v2(inputs, axis=1)
      else:
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        non_batch_dims = input_shape[1:]
        if non_batch_dims.is_fully_defined():
          last_dim = int(functools.reduce(operator.mul, non_batch_dims))
          flattened_shape = constant_op.constant([-1, last_dim])
        elif batch_dim is not None:
          flattened_shape = constant_op.constant([int(batch_dim), -1])
        else:
          flattened_shape = [array_ops.shape_v2(inputs)[0], -1]
        return array_ops.reshape(inputs, flattened_shape)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.as_shape(input_shape).as_list()
    if not input_shape:
      output_shape = tensor_shape.TensorShape([1])
    else:
      output_shape = [input_shape[0]]
    if np.all(input_shape[1:]):
      output_shape += [np.prod(input_shape[1:], dtype=int)]
    else:
      output_shape += [None]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Flatten, self).get_config()
    config.update({'data_format': self.data_format})
    return config

@keras_export('keras.layers.RepeatVector')
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

@keras_export('keras.layers.Lambda')
class Lambda(Layer):

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, output_shape=None, mask=None, arguments=None,
               **kwargs):
    super(Lambda, self).__init__(**kwargs)

    self.arguments = arguments or {}
    self.function = function

    if mask is not None:
      self.supports_masking = True
    self.mask = mask
    self._output_shape = output_shape

    self._already_warned = False

    function_args = tf_inspect.getfullargspec(function).args
    self._fn_expects_training_arg = 'training' in function_args
    self._fn_expects_mask_arg = 'mask' in function_args

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self._output_shape is None:
      with context.eager_mode():
        try:
          return super(Lambda, self).compute_output_shape(input_shape)
        except NotImplementedError:
          raise NotImplementedError(
              'We could not automatically infer the shape of the Lambda\'s '
              'output. Please specify `output_shape` for this Lambda.')

    if callable(self._output_shape):
      output_shapes = self._output_shape(input_shape)
      return tf_utils.convert_shapes(output_shapes, to_tuples=False)

    input_tensor_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    batch_size = nest.flatten(input_tensor_shape)[0][0] if input_shape else None

    def _add_batch(shape):
      return tensor_shape.TensorShape([batch_size] + shape.as_list())

    output_shapes = tf_utils.convert_shapes(self._output_shape, to_tuples=False)
    return nest.map_structure(_add_batch, output_shapes)

  def call(self, inputs, mask=None, training=None):
    kwargs = {k: v for k, v in self.arguments.items()}
    if self._fn_expects_mask_arg:
      kwargs['mask'] = mask
    if self._fn_expects_training_arg:
      kwargs['training'] = training

    created_variables = []
    def _variable_creator(next_creator, **kwargs):
      var = next_creator(**kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape,\
        variable_scope.variable_creator_scope(_variable_creator):
      result = self.function(inputs, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent(
      ).format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    return tf_logging.warn(msg)

  def compute_mask(self, inputs, mask=None):
    if callable(self.mask):
      return self.mask(inputs, mask)
    return self.mask

  def get_config(self):
    function_config = self._serialize_function_to_config(self.function)
    output_shape_config = self._serialize_function_to_config(self._output_shape,
                                                             allow_raw=True)
    config = {
        'function': function_config[0],
        'function_type': function_config[1],
        'module': function_config[2],
        'output_shape': output_shape_config[0],
        'output_shape_type': output_shape_config[1],
        'output_shape_module': output_shape_config[2],
    }
    if self.mask is not None:
      mask_config = self._serialize_function_to_config(self.mask)
      config.update({
          'mask': mask_config[0],
          'mask_type': mask_config[1],
          'mask_module': mask_config[2]
      })
    config['arguments'] = self.arguments

    base_config = super(Lambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _serialize_function_to_config(self, inputs, allow_raw=False):
    if isinstance(inputs, python_types.LambdaType):
      output = generic_utils.func_dump(inputs)
      output_type = 'lambda'
      module = inputs.__module__
    elif callable(inputs):
      output = inputs.__name__
      output_type = 'function'
      module = inputs.__module__
    elif allow_raw:
      output = inputs
      output_type = 'raw'
      module = None
    else:
      raise ValueError(
          'Invalid input for serialization, type: %s ' % type(inputs))

    return output, output_type, module

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    function = cls._parse_function_from_config(
        config, custom_objects, 'function', 'module', 'function_type')

    output_shape = cls._parse_function_from_config(
        config, custom_objects, 'output_shape', 'output_shape_module',
        'output_shape_type')
    if 'mask' in config:
      mask = cls._parse_function_from_config(
          config, custom_objects, 'mask', 'mask_module', 'mask_type')
    else:
      mask = None

    config['function'] = function
    config['output_shape'] = output_shape
    config['mask'] = mask

    if 'arguments' in config:
      for key in config['arguments']:
        if isinstance(config['arguments'][key], dict):
          arg_dict = config['arguments'][key]
          if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
            config['arguments'][key] = np.array(arg_dict['value'])

    return cls(**config)

  @classmethod
  def _parse_function_from_config(
      cls, config, custom_objects, func_attr_name, module_attr_name,
      func_type_attr_name):
    globs = globals().copy()
    module = config.pop(module_attr_name, None)
    if module in sys.modules:
      globs.update(sys.modules[module].__dict__)
    elif module is not None:
      warnings.warn('{} is not loaded, but a Lambda layer uses it. '
                    'It may cause errors.'.format(module)
                    , UserWarning)
    if custom_objects:
      globs.update(custom_objects)
    function_type = config.pop(func_type_attr_name)
    if function_type == 'function':
      function = generic_utils.deserialize_keras_object(
          config[func_attr_name],
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
      function = generic_utils.func_load(
          config[func_attr_name], globs=globs)
    elif function_type == 'raw':
      function = config[func_attr_name]
    else:
      raise TypeError('Unknown function type:', function_type)
    return function

@keras_export('keras.layers.Dense')
class Dense(Layer):

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
    super(Dense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    return core_ops.dense(
        inputs,
        self.kernel,
        self.bias,
        self.activation,
        dtype=self._compute_dtype_object)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    })
    return config

@keras_export('keras.layers.ActivityRegularization')
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

class TFOpLambda(Layer):

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    self.function = function
    self.symbol = (
        get_canonical_name_for_symbol(self.function,
                                      add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(self.function,
                                      api_name='keras',
                                      add_prefix_to_v1_names=True))
    kwargs['autocast'] = False

    def _call_wrapper(*args, **kwargs):
      return self._call_wrapper(*args, **kwargs)
    self.call = tf_decorator.make_decorator(function, _call_wrapper)

    super(TFOpLambda, self).__init__(**kwargs)

    self._already_warned = False

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def _call_wrapper(self, *args, **kwargs):
    created_variables = []
    def _variable_creator(next_creator, **creator_kwargs):
      var = next_creator(**creator_kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape, \
        variable_scope.variable_creator_scope(_variable_creator):
      kwargs.pop('name', None)
      result = self.function(*args, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent(
      ).format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    return tf_logging.warn(msg)

  def get_config(self):
    if not self.symbol:
      raise ValueError('This Keras op layer was generated from %s, a method '
                       'that is not an exposed in the TensorFlow API. This '
                       'may have happened if the method was explicitly '
                       'decorated to add dispatching support, and it was used '
                       'during Functional model construction. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.'
                       % self.function)
    config = {
        'function': self.symbol
    }

    base_config = super(TFOpLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config['function']
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError(
          'TF symbol `tf.%s` could not be found.' % symbol_name)

    config['function'] = function

    return cls(**config)

class KerasOpDispatcher(dispatch.GlobalOpDispatcher):

  def handle(self, op, args, kwargs):
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return TFOpLambda(op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

KerasOpDispatcher().register()
EOF
