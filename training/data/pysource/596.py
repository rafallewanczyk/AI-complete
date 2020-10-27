

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.Wrapper')
class Wrapper(Layer):

  def __init__(self, layer, **kwargs):
    assert isinstance(layer, Layer)
    self.layer = layer
    super(Wrapper, self).__init__(**kwargs)

  def build(self, input_shape=None):
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = True
    self.built = True

  @property
  def activity_regularizer(self):
    if hasattr(self.layer, 'activity_regularizer'):
      return self.layer.activity_regularizer
    else:
      return None

  def get_config(self):
    config = {
        'layer': {
            'class_name': self.layer.__class__.__name__,
            'config': self.layer.get_config()
        }
    }
    base_config = super(Wrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  
    config = config.copy()
    layer = deserialize_layer(
        config.pop('layer'), custom_objects=custom_objects)
    return cls(layer, **config)

@keras_export('keras.layers.TimeDistributed')
class TimeDistributed(Wrapper):

  def __init__(self, layer, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `TimeDistributed` layer with a '
          '`tf.keras.layers.Layer` instance. You passed: {input}'.format(
              input=layer))
    super(TimeDistributed, self).__init__(layer, **kwargs)
    self.supports_masking = True
    self._supports_ragged_inputs = True

    self._always_use_reshape = (
        layer_utils.is_builtin_layer(layer) and
        not getattr(layer, 'stateful', False))

  def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
    if int_shape is None:
      int_shape = K.int_shape(tensor)[start_idx:]
    if not any(not s for s in int_shape):
      return init_tuple + tuple(int_shape)
    shape = K.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
      if not s:
        int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) < 3:
      raise ValueError(
          '`TimeDistributed` Layer should be passed an `input_shape ` '
          'with at least 3 dimensions, received: ' + str(input_shape))
    self.input_spec = InputSpec(shape=[None, None] + input_shape[2:])
    child_input_shape = [input_shape[0]] + input_shape[2:]
    super(TimeDistributed, self).build(tuple(child_input_shape))
    self.built = True

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
                                                 input_shape[2:])
    child_output_shape = self.layer.compute_output_shape(child_input_shape)
    if not isinstance(child_output_shape, tensor_shape.TensorShape):
      child_output_shape = tensor_shape.TensorShape(child_output_shape)
    child_output_shape = child_output_shape.as_list()
    timesteps = input_shape[1]
    return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
                                    child_output_shape[1:])

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training

    input_shape = K.int_shape(inputs)
    if input_shape[0] and not self._always_use_reshape:
      inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
      is_ragged_input = row_lengths is not None

      def step(x, _):
        output = self.layer(x, **kwargs)
        return output, []

      _, outputs, _ = K.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=row_lengths[0] if is_ragged_input else input_shape[1],
          mask=mask,
          unroll=False)
      y = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
    else:
      if isinstance(inputs, ragged_tensor.RaggedTensor):
        y = self.layer(inputs.values, **kwargs)
        y = ragged_tensor.RaggedTensor.from_row_lengths(
            y,
            inputs.nested_row_lengths()[0])
      else:
        input_length = input_shape[1]
        if not input_length:
          input_length = array_ops.shape(inputs)[1]
        inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
        inputs = array_ops.reshape(inputs, inner_input_shape)
        if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
          inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
          kwargs['mask'] = K.reshape(mask, inner_mask_shape)

        y = self.layer(inputs, **kwargs)

        output_shape = self.compute_output_shape(input_shape).as_list()
        output_shape = self._get_shape_tuple((-1, input_length), y, 1,
                                             output_shape[2:])
        y = array_ops.reshape(y, output_shape)

    return y

  def compute_mask(self, inputs, mask=None):
    input_shape = K.int_shape(inputs)
    if input_shape[0] and not self._always_use_reshape or isinstance(
        inputs, ragged_tensor.RaggedTensor):
      return mask
    inner_mask = mask
    if inner_mask is not None:
      inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
      inner_mask = K.reshape(inner_mask, inner_mask_shape)
    inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
    inner_inputs = array_ops.reshape(inputs, inner_input_shape)
    output_mask = self.layer.compute_mask(inner_inputs, inner_mask)
    if output_mask is None:
      if mask is None:
        return None
      output_mask = mask
      for _ in range(2, len(K.int_shape(mask))):
        output_mask = K.any(output_mask, axis=-1)
    else:
      input_length = input_shape[1]
      if not input_length:
        input_length = K.shape(inputs)[1]
      output_mask_int_shape = K.int_shape(output_mask)
      if output_mask_int_shape is None:
        if mask is not None:
          output_mask_int_shape = K.int_shape(mask)
        else:
          output_mask_int_shape = K.compute_output_shape(input_shape)[:-1]
      output_mask_shape = self._get_shape_tuple(
          (-1, input_length), output_mask, 1, output_mask_int_shape[1:])
      output_mask = K.reshape(output_mask, output_mask_shape)
    return output_mask

@keras_export('keras.layers.Bidirectional')
class Bidirectional(Wrapper):

  def __init__(self,
               layer,
               merge_mode='concat',
               weights=None,
               backward_layer=None,
               **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `Bidirectional` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))
    if backward_layer is not None and not isinstance(backward_layer, Layer):
      raise ValueError('`backward_layer` need to be a `Layer` instance. '
                       'You passed: {input}'.format(input=backward_layer))
    if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
      raise ValueError('Invalid merge mode. '
                       'Merge mode should be one of '
                       '{"sum", "mul", "ave", "concat", None}')
    self._setattr_tracking = False
    super(Bidirectional, self).__init__(layer, **kwargs)
    self._setattr_tracking = True

    self.forward_layer = self._recreate_layer_from_config(layer)

    if backward_layer is None:
      self.backward_layer = self._recreate_layer_from_config(
          layer, go_backwards=True)
    else:
      self.backward_layer = backward_layer
      self._backward_layer_config = backward_layer.get_config()

    self.forward_layer._name = 'forward_' + self.forward_layer.name
    self.backward_layer._name = 'backward_' + self.backward_layer.name

    self._verify_layer_config()

    def force_zero_output_for_mask(layer):
      if getattr(layer, 'zero_output_for_mask', None) is not None:
        layer.zero_output_for_mask = layer.return_sequences

    force_zero_output_for_mask(self.forward_layer)
    force_zero_output_for_mask(self.backward_layer)

    self.merge_mode = merge_mode
    if weights:
      nw = len(weights)
      self.forward_layer.initial_weights = weights[:nw // 2]
      self.backward_layer.initial_weights = weights[nw // 2:]
    self.stateful = layer.stateful
    self.return_sequences = layer.return_sequences
    self.return_state = layer.return_state
    self.supports_masking = True
    self._trainable = True
    self._num_constants = 0
    self.input_spec = layer.input_spec
    self._supports_ragged_inputs = True

  def _verify_layer_config(self):
    if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
      raise ValueError('Forward layer and backward layer should have different '
                       '`go_backwards` value.')

    common_attributes = ('stateful', 'return_sequences', 'return_state')
    for a in common_attributes:
      forward_value = getattr(self.forward_layer, a)
      backward_value = getattr(self.backward_layer, a)
      if forward_value != backward_value:
        raise ValueError(
            'Forward layer and backward layer are expected to have the same '
            'value for attribute {attr}, got {forward} and {backward}'.format(
                attr=a, forward=forward_value, backward=backward_value))

  def _recreate_layer_from_config(self, layer, go_backwards=False):
    config = layer.get_config()
    if go_backwards:
      config['go_backwards'] = not config['go_backwards']
    if 'custom_objects' in tf_inspect.getfullargspec(
        layer.__class__.from_config).args:
      custom_objects = {}
      cell = getattr(layer, 'cell', None)
      if cell is not None:
        custom_objects[cell.__class__.__name__] = cell.__class__
        stacked_cells = getattr(cell, 'cells', [])
        for c in stacked_cells:
          custom_objects[c.__class__.__name__] = c.__class__
      return layer.__class__.from_config(config, custom_objects=custom_objects)
    else:
      return layer.__class__.from_config(config)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    output_shape = self.forward_layer.compute_output_shape(input_shape)
    if not isinstance(output_shape, tensor_shape.TensorShape):
      output_shape = tensor_shape.TensorShape(output_shape)
    output_shape = tuple(output_shape.as_list())
    if self.return_state:
      state_shape = output_shape[1:]
      output_shape = output_shape[0]

    if self.merge_mode == 'concat':
      output_shape = list(output_shape)
      output_shape[-1] *= 2
      output_shape = tuple(output_shape)
    elif self.merge_mode is None:
      output_shape = [output_shape, copy.copy(output_shape)]

    if self.return_state:
      if self.merge_mode is None:
        return output_shape + state_shape + copy.copy(state_shape)
      return [output_shape] + state_shape + copy.copy(state_shape)
    return output_shape

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    inputs, initial_state, constants = _standardize_args(
        inputs, initial_state, constants, self._num_constants)

    if isinstance(inputs, list):
      if len(inputs) > 1:
        initial_state = inputs[1:]
      inputs = inputs[0]

    if initial_state is None and constants is None:
      return super(Bidirectional, self).__call__(inputs, **kwargs)

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      num_states = len(initial_state)
      if num_states % 2 > 0:
        raise ValueError(
            'When passing `initial_state` to a Bidirectional RNN, '
            'the state should be a list containing the states of '
            'the underlying RNNs. '
            'Found: ' + str(initial_state))

      kwargs['initial_state'] = initial_state
      additional_inputs += initial_state
      state_specs = [InputSpec(shape=K.int_shape(state))
                     for state in initial_state]
      self.forward_layer.state_spec = state_specs[:num_states // 2]
      self.backward_layer.state_spec = state_specs[num_states // 2:]
      additional_specs += state_specs
    if constants is not None:
      kwargs['constants'] = constants
      additional_inputs += constants
      constants_spec = [InputSpec(shape=K.int_shape(constant))
                        for constant in constants]
      self.forward_layer.constants_spec = constants_spec
      self.backward_layer.constants_spec = constants_spec
      additional_specs += constants_spec

      self._num_constants = len(constants)
      self.forward_layer._num_constants = self._num_constants
      self.backward_layer._num_constants = self._num_constants

    is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
    for tensor in additional_inputs:
      if K.is_keras_tensor(tensor) != is_keras_tensor:
        raise ValueError('The initial state of a Bidirectional'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors'
                         ' (a "Keras tensor" is a tensor that was'
                         ' returned by a Keras layer, or by `Input`)')

    if is_keras_tensor:
      full_input = [inputs] + additional_inputs
      full_input_spec = [None for _ in range(len(nest.flatten(inputs)))
                        ] + additional_specs
      kwargs['initial_state'] = None
      kwargs['constants'] = None

      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(Bidirectional, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      return super(Bidirectional, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           training=None,
           mask=None,
           initial_state=None,
           constants=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training
    if generic_utils.has_arg(self.layer.call, 'mask'):
      kwargs['mask'] = mask
    if generic_utils.has_arg(self.layer.call, 'constants'):
      kwargs['constants'] = constants

    if generic_utils.has_arg(self.layer.call, 'initial_state'):
      if isinstance(inputs, list) and len(inputs) > 1:
        forward_inputs = [inputs[0]]
        backward_inputs = [inputs[0]]
        pivot = (len(inputs) - self._num_constants) // 2 + 1
        forward_inputs += inputs[1:pivot]
        if not self._num_constants:
          backward_inputs += inputs[pivot:]
        else:
          backward_inputs += inputs[pivot:-self._num_constants]
          forward_inputs += inputs[-self._num_constants:]
          backward_inputs += inputs[-self._num_constants:]
        forward_state, backward_state = None, None
        if 'constants' in kwargs:
          kwargs['constants'] = None
      elif initial_state is not None:
        forward_inputs, backward_inputs = inputs, inputs
        half = len(initial_state) // 2
        forward_state = initial_state[:half]
        backward_state = initial_state[half:]
      else:
        forward_inputs, backward_inputs = inputs, inputs
        forward_state, backward_state = None, None

      y = self.forward_layer(forward_inputs,
                             initial_state=forward_state, **kwargs)
      y_rev = self.backward_layer(backward_inputs,
                                  initial_state=backward_state, **kwargs)
    else:
      y = self.forward_layer(inputs, **kwargs)
      y_rev = self.backward_layer(inputs, **kwargs)

    if self.return_state:
      states = y[1:] + y_rev[1:]
      y = y[0]
      y_rev = y_rev[0]

    if self.return_sequences:
      y_rev = K.reverse(y_rev, 1)
    if self.merge_mode == 'concat':
      output = K.concatenate([y, y_rev])
    elif self.merge_mode == 'sum':
      output = y + y_rev
    elif self.merge_mode == 'ave':
      output = (y + y_rev) / 2
    elif self.merge_mode == 'mul':
      output = y * y_rev
    elif self.merge_mode is None:
      output = [y, y_rev]
    else:
      raise ValueError(
          'Unrecognized value for `merge_mode`: %s' % (self.merge_mode))

    if self.return_state:
      if self.merge_mode is None:
        return output + states
      return [output] + states
    return output

  def reset_states(self):
    self.forward_layer.reset_states()
    self.backward_layer.reset_states()

  def build(self, input_shape):
    with K.name_scope(self.forward_layer.name):
      self.forward_layer.build(input_shape)
    with K.name_scope(self.backward_layer.name):
      self.backward_layer.build(input_shape)
    self.built = True

  def compute_mask(self, inputs, mask):
    if isinstance(mask, list):
      mask = mask[0]
    if self.return_sequences:
      if not self.merge_mode:
        output_mask = [mask, mask]
      else:
        output_mask = mask
    else:
      output_mask = [None, None] if not self.merge_mode else None

    if self.return_state:
      states = self.forward_layer.states
      state_mask = [None for _ in states]
      if isinstance(output_mask, list):
        return output_mask + state_mask * 2
      return [output_mask] + state_mask * 2
    return output_mask

  @property
  def constraints(self):
    constraints = {}
    if hasattr(self.forward_layer, 'constraints'):
      constraints.update(self.forward_layer.constraints)
      constraints.update(self.backward_layer.constraints)
    return constraints

  def get_config(self):
    config = {'merge_mode': self.merge_mode}
    if self._num_constants:
      config['num_constants'] = self._num_constants

    if hasattr(self, '_backward_layer_config'):
      config['backward_layer'] = {
          'class_name': self.backward_layer.__class__.__name__,
          'config': self._backward_layer_config,
      }
    base_config = super(Bidirectional, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    num_constants = config.pop('num_constants', 0)
    backward_layer_config = config.pop('backward_layer', None)
    if backward_layer_config is not None:
      from tensorflow.python.keras.layers import deserialize as deserialize_layer  
      backward_layer = deserialize_layer(
          backward_layer_config, custom_objects=custom_objects)
      config['backward_layer'] = backward_layer

    layer = super(Bidirectional, cls).from_config(config,
                                                  custom_objects=custom_objects)
    layer._num_constants = num_constants
    return layer
EOF
