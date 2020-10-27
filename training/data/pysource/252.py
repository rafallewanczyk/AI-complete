

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export

class _Merge(Layer):

  def __init__(self, **kwargs):
    super(_Merge, self).__init__(**kwargs)
    self.supports_masking = True

  def _merge_function(self, inputs):
    raise NotImplementedError

  def _compute_elemwise_op_output_shape(self, shape1, shape2):
    if None in [shape1, shape2]:
      return None
    elif len(shape1) < len(shape2):
      return self._compute_elemwise_op_output_shape(shape2, shape1)
    elif not shape2:
      return shape1
    output_shape = list(shape1[:-len(shape2)])
    for i, j in zip(shape1[-len(shape2):], shape2):
      if i is None or j is None:
        output_shape.append(None)
      elif i == 1:
        output_shape.append(j)
      elif j == 1:
        output_shape.append(i)
      else:
        if i != j:
          raise ValueError(
              'Operands could not be broadcast '
              'together with shapes ' + str(shape1) + ' ' + str(shape2))
        output_shape.append(i)
    return tuple(output_shape)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if not isinstance(input_shape[0], tuple):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if len(input_shape) < 2:
      raise ValueError('A merge layer should be called '
                       'on a list of at least 2 inputs. '
                       'Got ' + str(len(input_shape)) + ' inputs.')
    batch_sizes = {s[0] for s in input_shape if s} - {None}
    if len(batch_sizes) > 1:
      raise ValueError(
          'Can not merge tensors with different '
          'batch sizes. Got tensors with shapes : ' + str(input_shape))
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    if None not in input_shape and len(set(map(len, input_shape))) == 1:
      self._reshape_required = False
    else:
      self._reshape_required = True

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise ValueError('A merge layer should be called on a list of inputs.')
    if self._reshape_required:
      reshaped_inputs = []
      input_ndims = list(map(K.ndim, inputs))
      if None not in input_ndims:
        max_ndim = max(input_ndims)
        for x in inputs:
          x_ndim = K.ndim(x)
          for _ in range(max_ndim - x_ndim):
            x = array_ops.expand_dims(x, axis=1)
          reshaped_inputs.append(x)
        return self._merge_function(reshaped_inputs)
      else:
        transposed = False
        for x in inputs:
          x_ndim = K.ndim(x)
          if x_ndim is None:
            x_shape = array_ops.shape(x)
            batch_size = x_shape[0]
            new_shape = K.concatenate(
                [x_shape[1:],
                 array_ops.expand_dims(batch_size, axis=-1)])
            x_transposed = array_ops.reshape(
                x,
                array_ops.stack(
                    [batch_size, math_ops.reduce_prod(x_shape[1:])], axis=0))
            x_transposed = array_ops.transpose(x_transposed, perm=(1, 0))
            x_transposed = array_ops.reshape(x_transposed, new_shape)
            reshaped_inputs.append(x_transposed)
            transposed = True
          elif x_ndim > 1:
            dims = list(range(1, x_ndim)) + [0]
            reshaped_inputs.append(array_ops.transpose(x, perm=dims))
            transposed = True
          else:
            reshaped_inputs.append(x)
        y = self._merge_function(reshaped_inputs)
        y_ndim = K.ndim(y)
        if transposed:
          if y_ndim is None:
            y_shape = array_ops.shape(y)
            y_ndim = array_ops.shape(y_shape)[0]
            batch_size = y_shape[y_ndim - 1]
            new_shape = K.concatenate([
                array_ops.expand_dims(batch_size, axis=-1), y_shape[:y_ndim - 1]
            ])
            y = array_ops.reshape(y, (-1, batch_size))
            y = array_ops.transpose(y, perm=(1, 0))
            y = array_ops.reshape(y, new_shape)
          elif y_ndim > 1:
            dims = [y_ndim - 1] + list(range(y_ndim - 1))
            y = array_ops.transpose(y, perm=dims)
        return y
    else:
      return self._merge_function(inputs)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if input_shape[0] is None:
      output_shape = None
    else:
      output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
      if input_shape[i] is None:
        shape = None
      else:
        shape = input_shape[i][1:]
      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
    if len(batch_sizes) == 1:
      output_shape = (list(batch_sizes)[0],) + output_shape
    else:
      output_shape = (None,) + output_shape
    return output_shape

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, (tuple, list)):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, (tuple, list)):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    masks = [array_ops.expand_dims(m, axis=0) for m in mask if m is not None]
    return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)

@keras_export('keras.layers.Add')
class Add(_Merge):

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output

@keras_export('keras.layers.Subtract')
class Subtract(_Merge):

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    super(Subtract, self).build(input_shape)
    if len(input_shape) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')

  def _merge_function(self, inputs):
    if len(inputs) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')
    return inputs[0] - inputs[1]

@keras_export('keras.layers.Multiply')
class Multiply(_Merge):

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output *= inputs[i]
    return output

@keras_export('keras.layers.Average')
class Average(_Merge):

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output / len(inputs)

@keras_export('keras.layers.Maximum')
class Maximum(_Merge):

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = math_ops.maximum(output, inputs[i])
    return output

@keras_export('keras.layers.Minimum')
class Minimum(_Merge):

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = math_ops.minimum(output, inputs[i])
    return output

@keras_export('keras.layers.Concatenate')
class Concatenate(_Merge):

  def __init__(self, axis=-1, **kwargs):
    super(Concatenate, self).__init__(**kwargs)
    self.axis = axis
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if not isinstance(input_shape[0], tuple) or len(input_shape) < 2:
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of at least 2 inputs')
    if all(shape is None for shape in input_shape):
      return
    reduced_inputs_shapes = [list(shape) for shape in input_shape]
    shape_set = set()
    for i in range(len(reduced_inputs_shapes)):
      del reduced_inputs_shapes[i][self.axis]
      shape_set.add(tuple(reduced_inputs_shapes[i]))

    if len(shape_set) != 1:
      err_msg = ('A `Concatenate` layer requires inputs with matching shapes '
                 'except for the concat axis. Got inputs shapes: %s' %
                 input_shape)
      ranks = set(len(shape) for shape in shape_set)
      if len(ranks) != 1:
        raise ValueError(err_msg)
      (rank,) = ranks
      for axis in range(rank):
        unique_dims = set(
            shape[axis] for shape in shape_set if shape[axis] is not None)
        if len(unique_dims) > 1:
          raise ValueError(err_msg)

  def _merge_function(self, inputs):
    return K.concatenate(inputs, axis=self.axis)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, (tuple, list)):
      raise ValueError('A `Concatenate` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    output_shape = list(input_shapes[0])
    for shape in input_shapes[1:]:
      if output_shape[self.axis] is None or shape[self.axis] is None:
        output_shape[self.axis] = None
        break
      output_shape[self.axis] += shape[self.axis]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      return None
    if not isinstance(mask, (tuple, list)):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, (tuple, list)):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all(m is None for m in mask):
      return None
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        masks.append(array_ops.ones_like(input_i, dtype='bool'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        masks.append(array_ops.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)
    concatenated = K.concatenate(masks, axis=self.axis)
    return K.all(concatenated, axis=-1, keepdims=False)

  def get_config(self):
    config = {
        'axis': self.axis,
    }
    base_config = super(Concatenate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.layers.Dot')
class Dot(_Merge):

  def __init__(self, axes, normalize=False, **kwargs):
    super(Dot, self).__init__(**kwargs)
    if not isinstance(axes, int):
      if not isinstance(axes, (list, tuple)):
        raise TypeError('Invalid type for `axes` - '
                        'should be a list or an int.')
      if len(axes) != 2:
        raise ValueError('Invalid format for `axes` - '
                         'should contain two elements.')
      if not isinstance(axes[0], int) or not isinstance(axes[1], int):
        raise ValueError('Invalid format for `axes` - '
                         'list elements should be "int".')
    self.axes = axes
    self.normalize = normalize
    self.supports_masking = True
    self._reshape_required = False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if not isinstance(input_shape[0], tuple) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = input_shape[0]
    shape2 = input_shape[1]
    if shape1 is None or shape2 is None:
      return
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    if shape1[axes[0]] != shape2[axes[1]]:
      raise ValueError('Dimension incompatibility '
                       '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                       'Layer shapes: %s, %s. ' % (shape1, shape2) +
                       'Chosen axes: %s, %s' % (axes[0], axes[1]))

  def _merge_function(self, inputs):
    base_layer_utils.no_ragged_support(inputs, self.name)
    if len(inputs) != 2:
      raise ValueError('A `Dot` layer should be called on exactly 2 inputs')
    x1 = inputs[0]
    x2 = inputs[1]
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = []
      for i in range(len(self.axes)):
        if self.axes[i] < 0:
          axes.append(self.axes[i] % K.ndim(inputs[i]))
        else:
          axes.append(self.axes[i])
    if self.normalize:
      x1 = nn.l2_normalize(x1, axis=axes[0])
      x2 = nn.l2_normalize(x2, axis=axes[1])
    output = K.batch_dot(x1, x2, axes)
    return output

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
      raise ValueError('A `Dot` layer should be called '
                       'on a list of 2 inputs.')
    shape1 = list(input_shape[0])
    shape2 = list(input_shape[1])
    if isinstance(self.axes, int):
      if self.axes < 0:
        axes = [self.axes % len(shape1), self.axes % len(shape2)]
      else:
        axes = [self.axes] * 2
    else:
      axes = self.axes
    shape1.pop(axes[0])
    shape2.pop(axes[1])
    shape2.pop(0)
    output_shape = shape1 + shape2
    if len(output_shape) == 1:
      output_shape += [1]
    return tuple(output_shape)

  def compute_mask(self, inputs, mask=None):
    return None

  def get_config(self):
    config = {
        'axes': self.axes,
        'normalize': self.normalize,
    }
    base_config = super(Dot, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.layers.add')
def add(inputs, **kwargs):
  return Add(**kwargs)(inputs)

@keras_export('keras.layers.subtract')
def subtract(inputs, **kwargs):
  return Subtract(**kwargs)(inputs)

@keras_export('keras.layers.multiply')
def multiply(inputs, **kwargs):
  return Multiply(**kwargs)(inputs)

@keras_export('keras.layers.average')
def average(inputs, **kwargs):
  return Average(**kwargs)(inputs)

@keras_export('keras.layers.maximum')
def maximum(inputs, **kwargs):
  return Maximum(**kwargs)(inputs)

@keras_export('keras.layers.minimum')
def minimum(inputs, **kwargs):
  return Minimum(**kwargs)(inputs)

@keras_export('keras.layers.concatenate')
def concatenate(inputs, axis=-1, **kwargs):
  return Concatenate(axis=axis, **kwargs)(inputs)

@keras_export('keras.layers.dot')
def dot(inputs, axes, normalize=False, **kwargs):
  return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)
EOF
