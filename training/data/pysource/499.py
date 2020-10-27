

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import re

import numpy as np
from six.moves import zip  

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.contrib.keras.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.contrib.keras.python.keras.utils.layer_utils import print_summary as print_layer_summary
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.platform import tf_logging as logging

try:
  import h5py
except ImportError:
  h5py = None

try:
  import yaml
except ImportError:
  yaml = None

InputSpec = tf_base_layers.InputSpec
Node = tf_base_layers.Node
TFBaseLayer = tf_base_layers.Layer

class Layer(tf_base_layers.Layer):

  def __init__(self, **kwargs):
    allowed_kwargs = {
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'dtype',
        'name',
        'trainable',
        'weights',
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    name = kwargs.get('name')

    trainable = kwargs.get('trainable', True)

    dtype = kwargs.get('dtype')
    if dtype is None:
      dtype = K.floatx()

    super(Layer, self).__init__(name=name, dtype=dtype, trainable=trainable)

    self.supports_masking = False

    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self.batch_input_shape = batch_input_shape

    if 'weights' in kwargs:
      self._initial_weights = kwargs['weights']
    else:
      self._initial_weights = None

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 constraint=None):
    if dtype is None:
      dtype = K.floatx()
    weight = self.add_variable(name, shape,
                               dtype=dtype,
                               initializer=initializer,
                               regularizer=regularizer,
                               constraint=constraint,
                               trainable=trainable)
    return weight

  def call(self, inputs, **kwargs):  
    return inputs

  def __call__(self, inputs, **kwargs):
    output = super(Layer, self).__call__(inputs, **kwargs)

    output_tensors = _to_list(output)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False) for x in _to_list(inputs)])
    uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
    for i in range(len(output_tensors)):
      output_tensors[i]._uses_learning_phase = getattr(
          output_tensors[i], '_uses_learning_phase', False) or uses_lp

    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return output

  def _compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      return [tensor_shape.TensorShape(shape) for shape in input_shape]
    else:
      return tensor_shape.TensorShape(input_shape)

  def compute_mask(self, inputs, mask=None):  
    if not self.supports_masking:
      if mask is not None:
        if isinstance(mask, list):
          if any(m is not None for m in mask):
            raise TypeError('Layer ' + self.name + ' does not support masking, '
                            'but was passed an input_mask: ' + str(mask))
        else:
          raise TypeError('Layer ' + self.name + ' does not support masking, '
                          'but was passed an input_mask: ' + str(mask))
      return None
    return mask

  def get_input_mask_at(self, node_index):
    inputs = self.get_input_at(node_index)
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  def get_output_mask_at(self, node_index):
    output = self.get_output_at(node_index)
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @property
  def input_mask(self):
    inputs = self.input
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @property
  def output_mask(self):
    output = self.output
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  def set_weights(self, weights):
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('You called `set_weights(weights)` on layer "' +
                       self.name + '" with a  weight list of length ' +
                       str(len(weights)) + ', but the layer was expecting ' +
                       str(len(params)) + ' weights. Provided weights: ' +
                       str(weights)[:50] + '...')
    if not params:
      return
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Layer weight shape ' + str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    params = self.weights
    return K.batch_get_value(params)

  def get_config(self):
    config = {'name': self.name, 'trainable': self.trainable}
    if hasattr(self, 'batch_input_shape'):
      config['batch_input_shape'] = self.batch_input_shape
    if hasattr(self, 'dtype'):
      config['dtype'] = self.dtype
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)

class InputLayer(tf_base_layers.InputLayer, Layer):

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=False,
               name=None,
               **kwargs):
    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(K.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = K.floatx()
      else:
        dtype = K.dtype(input_tensor)
    super(InputLayer, self).__init__(input_shape=input_shape,
                                     batch_size=batch_size,
                                     dtype=dtype,
                                     input_tensor=input_tensor,
                                     sparse=sparse,
                                     name=name)

  def get_config(self):
    config = {
        'batch_input_shape': self.batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config

def Input(  
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs):
  if 'batch_shape' in kwargs:
    batch_shape = kwargs.pop('batch_shape')
    if shape and batch_shape:
      raise ValueError('Only provide the shape OR '
                       'batch_shape argument to '
                       'Input, not both at the same time.')
    batch_size = batch_shape[0]
    shape = batch_shape[1:]
  if kwargs:
    raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

  if dtype is None:
    dtype = K.floatx()
  if not shape and tensor is None:
    raise ValueError('Please provide to Input either a `shape`'
                     ' or a `tensor` argument. Note that '
                     '`shape` does not include the batch '
                     'dimension.')
  input_layer = InputLayer(
      input_shape=shape,
      batch_size=batch_size,
      name=name,
      dtype=dtype,
      sparse=sparse,
      input_tensor=tensor)
  outputs = input_layer.inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs

class Network(tf_base_layers.Network, Layer):

  def __init__(self, inputs, outputs, name=None):
    super(Network, self).__init__(inputs, outputs, name=name)

    self.supports_masking = False
    masks = []
    for x in self.inputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None
      masks.append(mask)
    mask_cache_key = (tf_base_layers._object_list_uid(self.inputs) + '_' +
                      tf_base_layers._object_list_uid(masks))
    masks = []
    for x in self.outputs:
      mask = x._keras_mask if hasattr(x, '_keras_mask') else None
      masks.append(mask)
    if len(masks) == 1:
      mask = masks[0]
    else:
      mask = masks
    self._output_mask_cache[mask_cache_key] = mask

    self.input_names = []
    self.output_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for i, layer in enumerate(self._input_layers):
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        self._feed_inputs.append(layer.input)
        self._feed_input_shapes.append(K.int_shape(self.inputs[i]))
    for layer in self._output_layers:
      self.output_names.append(layer.name)

    self.internal_input_shapes = [K.int_shape(x) for x in self.inputs]
    self.internal_output_shapes = [K.int_shape(x) for x in self.outputs]

  @property
  def uses_learning_phase(self):
    return any(
        [getattr(x, '_uses_learning_phase', False) for x in self.outputs])

  @property
  def stateful(self):
    return any([(hasattr(layer, 'stateful') and layer.stateful)
                for layer in self.layers])

  def reset_states(self):
    for layer in self.layers:
      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
        layer.reset_states()

  @property
  def state_updates(self):
    state_updates = []
    for layer in self.layers:
      if getattr(layer, 'stateful', False):
        if hasattr(layer, 'updates'):
          state_updates += layer.updates
    return state_updates

  @property
  def constraints(self):
    cons = {}
    for layer in self.layers:
      for key, value in layer.constraints.items():
        if key in cons and cons[key] != value:
          raise ValueError('Received multiple constraints '
                           'for one weight tensor: ' + str(key))
        cons[key] = value
    return cons

  def get_weights(self):
    weights = []
    for layer in self.layers:
      weights += layer.weights
    return K.batch_get_value(weights)

  def set_weights(self, weights):
    tuples = []
    for layer in self.layers:
      num_param = len(layer.weights)
      layer_weights = weights[:num_param]
      for sw, w in zip(layer.weights, layer_weights):
        tuples.append((sw, w))
      weights = weights[num_param:]
    K.batch_set_value(tuples)

  def compute_mask(self, inputs, mask):
    inputs = _to_list(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = _to_list(mask)
    cache_key = ','.join([str(id(x)) for x in inputs])
    cache_key += '_' + ','.join([str(id(x)) for x in masks])
    if cache_key in self._output_mask_cache:
      return self._output_mask_cache[cache_key]
    else:
      _, output_masks, _ = self._run_internal_graph(inputs, masks)
      return output_masks

  def get_config(self):
    config = {
        'name': self.name,
    }
    node_conversion_map = {}
    for layer in self.layers:
      if issubclass(layer.__class__, Network):
        kept_nodes = 1
      else:
        kept_nodes = 0
      for original_node_index, node in enumerate(layer.inbound_nodes):
        node_key = tf_base_layers._make_node_key(layer.name,
                                                 original_node_index)
        if node_key in self._network_nodes:
          node_conversion_map[node_key] = kept_nodes
          kept_nodes += 1
    layer_configs = []
    for layer in self.layers:  
      layer_class_name = layer.__class__.__name__
      layer_config = layer.get_config()
      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer.inbound_nodes):
        node_key = tf_base_layers._make_node_key(layer.name,
                                                 original_node_index)
        if node_key in self._network_nodes:
          if node.arguments:
            try:
              json.dumps(node.arguments)
              kwargs = node.arguments
            except TypeError:
              logging.warning(
                  'Layer ' + layer.name +
                  ' was passed non-serializable keyword arguments: ' +
                  str(node.arguments) + '. They will not be included '
                  'in the serialized model (and thus will be missing '
                  'at deserialization time).')
              kwargs = {}
          else:
            kwargs = {}
          if node.inbound_layers:
            node_data = []
            for i in range(len(node.inbound_layers)):
              inbound_layer = node.inbound_layers[i]
              node_index = node.node_indices[i]
              tensor_index = node.tensor_indices[i]
              node_key = tf_base_layers._make_node_key(inbound_layer.name,
                                                       node_index)
              new_node_index = node_conversion_map.get(node_key, 0)
              node_data.append(
                  [inbound_layer.name, new_node_index, tensor_index, kwargs])
            filtered_inbound_nodes.append(node_data)
      layer_configs.append({
          'name': layer.name,
          'class_name': layer_class_name,
          'config': layer_config,
          'inbound_nodes': filtered_inbound_nodes,
      })
    config['layers'] = layer_configs

    model_inputs = []
    for i in range(len(self._input_layers)):
      layer, node_index, tensor_index = self._input_coordinates[i]
      node_key = tf_base_layers._make_node_key(layer.name,
                                               node_index)
      new_node_index = node_conversion_map[node_key]
      model_inputs.append([layer.name, new_node_index, tensor_index])
    config['input_layers'] = model_inputs
    model_outputs = []
    for i in range(len(self._output_layers)):
      layer, node_index, tensor_index = self._output_coordinates[i]
      node_key = tf_base_layers._make_node_key(layer.name,
                                               node_index)
      new_node_index = node_conversion_map[node_key]
      model_outputs.append([layer.name, new_node_index, tensor_index])
    config['output_layers'] = model_outputs
    return copy.deepcopy(config)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    created_layers = {}

    def process_layer(layer_data):
      layer_name = layer_data['name']

      from tensorflow.contrib.keras.python.keras.layers import deserialize as deserialize_layer  
      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

      inbound_nodes_data = layer_data['inbound_nodes']
      for node_data in inbound_nodes_data:
        input_tensors = []
        for input_data in node_data:
          inbound_layer_name = input_data[0]
          inbound_node_index = input_data[1]
          inbound_tensor_index = input_data[2]
          if len(input_data) == 3:
            kwargs = {}
          elif len(input_data) == 4:
            kwargs = input_data[3]
          else:
            raise ValueError('Improperly formatted model config.')
          if inbound_layer_name not in created_layers:
            raise ValueError('Missing layer: ' + inbound_layer_name)
          inbound_layer = created_layers[inbound_layer_name]
          inbound_node = inbound_layer.inbound_nodes[inbound_node_index]
          input_tensors.append(
              inbound_node.output_tensors[inbound_tensor_index])
        if input_tensors:
          if len(input_tensors) == 1:
            layer(input_tensors[0], **kwargs)
          else:
            layer(input_tensors, **kwargs)

    for layer_data in config['layers']:
      process_layer(layer_data)

    name = config.get('name')
    input_tensors = []
    output_tensors = []
    for layer_data in config['input_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
      input_tensors.append(layer_output_tensors[tensor_index])
    for layer_data in config['output_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
      output_tensors.append(layer_output_tensors[tensor_index])
    return cls(inputs=input_tensors, outputs=output_tensors, name=name)

  def save(self, filepath, overwrite=True, include_optimizer=True):
    from tensorflow.contrib.keras.python.keras.models import save_model  
    save_model(self, filepath, overwrite, include_optimizer)

  def save_weights(self, filepath, overwrite=True):
    if h5py is None:
      raise ImportError('`save_weights` requires h5py.')
    if not overwrite and os.path.isfile(filepath):
      proceed = ask_to_proceed_with_overwrite(filepath)
      if not proceed:
        return
    f = h5py.File(filepath, 'w')
    save_weights_to_hdf5_group(f, self.layers)
    f.flush()
    f.close()

  def load_weights(self, filepath, by_name=False):
    if h5py is None:
      raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
      f = f['model_weights']
    if by_name:
      load_weights_from_hdf5_group_by_name(f, self.layers)
    else:
      load_weights_from_hdf5_group(f, self.layers)

    if hasattr(f, 'close'):
      f.close()

  def _updated_config(self):
    from tensorflow.contrib.keras.python.keras import __version__ as keras_version  

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': K.backend()
    }
    return model_config

  def to_json(self, **kwargs):

    def get_json_type(obj):
      if type(obj).__module__ == np.__name__:
        return obj.item()

      if type(obj).__name__ == type.__name__:
        return obj.__name__

      raise TypeError('Not JSON Serializable:', obj)

    model_config = self._updated_config()
    return json.dumps(model_config, default=get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    if yaml is None:
      raise ImportError('Requires yaml module installed.')
    return yaml.dump(self._updated_config(), **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    print_layer_summary(self,
                        line_length=line_length,
                        positions=positions,
                        print_fn=print_fn)

Container = Network

def get_source_inputs(tensor, layer=None, node_index=None):
  if not hasattr(tensor, '_keras_history'):
    return tensor

  if layer is None or node_index:
    layer, node_index, _ = tensor._keras_history
  if not layer.inbound_nodes:
    return [tensor]
  else:
    node = layer.inbound_nodes[node_index]
    if not node.inbound_layers:
      return node.input_tensors
    else:
      source_tensors = []
      for i in range(len(node.inbound_layers)):
        x = node.input_tensors[i]
        layer = node.inbound_layers[i]
        node_index = node.node_indices[i]
        previous_sources = get_source_inputs(x, layer, node_index)
        for x in previous_sources:
          if x not in source_tensors:
            source_tensors.append(x)
      return source_tensors

def _to_list(x):
  if isinstance(x, list):
    return x
  return [x]

def _object_list_uid(object_list):
  object_list = _to_list(object_list)
  return ', '.join([str(abs(id(x))) for x in object_list])

def _to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure

def _collect_input_shape(input_tensors):
  input_tensors = _to_list(input_tensors)
  shapes = []
  for x in input_tensors:
    shapes.append(K.int_shape(x))
  if len(shapes) == 1:
    return shapes[0]
  return shapes

def save_weights_to_hdf5_group(f, layers):
  from tensorflow.contrib.keras.python.keras import __version__ as keras_version  

  f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
  f.attrs['backend'] = K.backend().encode('utf8')
  f.attrs['keras_version'] = str(keras_version).encode('utf8')

  for layer in layers:
    g = f.create_group(layer.name)
    symbolic_weights = layer.weights
    weight_values = K.batch_get_value(symbolic_weights)
    weight_names = []
    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
      if hasattr(w, 'name') and w.name:
        name = str(w.name)
      else:
        name = 'param_' + str(i)
      weight_names.append(name.encode('utf8'))
    g.attrs['weight_names'] = weight_names
    for name, val in zip(weight_names, weight_values):
      param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
      if not val.shape:
        param_dset[()] = val
      else:
        param_dset[:] = val

def preprocess_weights_for_loading(layer,
                                   weights,
                                   original_keras_version=None,
                                   original_backend=None):
  if original_keras_version == '1':
    if layer.__class__.__name__ == 'Bidirectional':
      num_weights_per_layer = len(weights) // 2

      forward_weights = preprocess_weights_for_loading(
          layer.forward_layer, weights[:num_weights_per_layer],
          original_keras_version, original_backend)
      backward_weights = preprocess_weights_for_loading(
          layer.backward_layer, weights[num_weights_per_layer:],
          original_keras_version, original_backend)
      weights = forward_weights + backward_weights

    if layer.__class__.__name__ == 'TimeDistributed':
      weights = preprocess_weights_for_loading(
          layer.layer, weights, original_keras_version, original_backend)

    if layer.__class__.__name__ == 'Conv1D':
      shape = weights[0].shape
      if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
        assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0],
                                                           1)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
      weights[0] = weights[0][:, 0, :, :]

    if layer.__class__.__name__ == 'Conv2D':
      if layer.data_format == 'channels_first':
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

    if layer.__class__.__name__ == 'Conv2DTranspose':
      if layer.data_format == 'channels_last':
        weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
      if layer.data_format == 'channels_first':
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

    if layer.__class__.__name__ == 'Conv3D':
      if layer.data_format == 'channels_first':
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

    if layer.__class__.__name__ == 'GRU':
      if len(weights) == 9:
        kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[4], weights[7]], axis=-1)
        bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'LSTM':
      if len(weights) == 12:
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'ConvLSTM2D':
      if len(weights) == 12:
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        if layer.data_format == 'channels_first':
          kernel = np.transpose(kernel, (2, 3, 1, 0))
          recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ in ['Model', 'Sequential']:
      new_weights = []
      for sublayer in layer.layers:
        num_weights = len(sublayer.trainable_weights)
        if num_weights > 0:
          new_weights.extend(
              preprocess_weights_for_loading(
                  layer=sublayer,
                  weights=weights[:num_weights],
                  original_keras_version=original_keras_version,
                  original_backend=original_backend))
          weights = weights[num_weights:]

      for sublayer in layer.layers:
        num_weights = len([
            l for l in sublayer.weights if l not in sublayer.trainable_weights
        ])
        if num_weights > 0:
          new_weights.extend(
              preprocess_weights_for_loading(
                  layer=sublayer,
                  weights=weights[:num_weights],
                  original_keras_version=original_keras_version,
                  original_backend=original_backend))
          weights = weights[num_weights:]
      weights = new_weights

  conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
  if layer.__class__.__name__ in conv_layers:
    if original_backend and K.backend() != original_backend:
      weights[0] = conv_utils.convert_kernel(weights[0])
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = conv_utils.convert_kernel(weights[1])
    if K.int_shape(layer.weights[0]) != weights[0].shape:
      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))
  return weights

def load_weights_from_hdf5_group(f, layers):
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  filtered_layers = []
  for layer in layers:
    weights = layer.weights
    if weights:
      filtered_layers.append(layer)

  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != len(filtered_layers):
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(len(filtered_layers)) +
                     ' layers.')

  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = layer.weights
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer 
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)

def load_weights_from_hdf5_group_by_name(f, layers):
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

  index = {}
  for layer in layers:
    if layer.name:
      index.setdefault(layer.name, []).append(layer)

  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]

    for layer in index.get(name, []):
      symbolic_weights = layer.weights
      weight_values = preprocess_weights_for_loading(
          layer, weight_values, original_keras_version, original_backend)
      if len(weight_values) != len(symbolic_weights):
        raise ValueError('Layer 
                         '") expects ' + str(len(symbolic_weights)) +
                         ' weight(s), but the saved weights' + ' have ' +
                         str(len(weight_values)) + ' element(s).')
      for i in range(len(weight_values)):
        weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
  K.batch_set_value(weight_value_tuples)
EOF
