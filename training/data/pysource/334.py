

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import json
import os
import threading

import numpy as np
from six.moves import zip  

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import saving
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import serialization
from tensorflow.python.util import tf_inspect

try:
  import h5py
except ImportError:
  h5py = None

try:
  import yaml
except ImportError:
  yaml = None

class Network(base_layer.Layer):

  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs',),
      base_layer.Layer._TF_MODULE_IGNORED_PROPERTIES
  ))

  def __init__(self, *args, **kwargs):  
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      self._init_graph_network(*args, **kwargs)
    else:
      self._init_subclassed_network(**kwargs)

    tf_utils.assert_no_legacy_layers(self.layers)

  @trackable.no_automatic_dependency_tracking
  def _base_init(self, name=None, **kwargs):

    generic_utils.validate_kwargs(kwargs, {'trainable', 'dtype', 'dynamic',
                                           'autocast'})

    self._thread_local = threading.local()

    self._init_set_name(name, zero_based=True)
    self._activity_regularizer = None
    self._trainable = kwargs.get('trainable', True)
    self._dynamic = kwargs.get('dynamic', False)
    self._is_compiled = False
    self._layers = []

    self._compute_output_and_mask_jointly = False

    self.supports_masking = False
    if not hasattr(self, 'optimizer'):
      self.optimizer = None

    self._maybe_create_attribute('_trainable_weights', [])
    self._maybe_create_attribute('_non_trainable_weights', [])
    self._updates = []  
    self._losses = []
    self._callable_losses = []
    self._metrics = []
    self._scope = None  
    self._reuse = None  
    if context.executing_eagerly():
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  

    self._set_dtype_policy(kwargs.get('dtype', None))

    self._maybe_create_attribute('_layers', [])

    self._outbound_nodes = []
    self._inbound_nodes = []

    self._trackable_saver = (
        trackable_utils.saver_with_op_caching(self))

  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {'trainable'},
        'Functional models may only specify `name` and `trainable` keyword '
        'arguments during initialization. Got an unexpected argument:')
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(name=name, **kwargs)
    self._validate_graph_inputs_and_outputs()

    self.built = True
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True
    self._expects_training_arg = True
    self._expects_mask_arg = True
    self._autocast = False

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

    self._track_layers(layers)

    node_module.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self._nested_inputs,
        output_tensors=self._nested_outputs)

    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for i, layer in enumerate(self._input_layers):
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)

  def _set_output_names(self):
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified

  @trackable.no_automatic_dependency_tracking
  def _init_subclassed_network(self, name=None, **kwargs):
    self._base_init(name=name, **kwargs)
    self._is_graph_network = False
    self._init_call_fn_args()
    self._autocast = kwargs.get('autocast',
                                base_layer_utils.v2_dtype_behavior_enabled())
    self.outputs = []
    self.inputs = []
    self.built = False

  @property
  def dynamic(self):
    if self._is_graph_network:
      return any(layer.dynamic for layer in self.layers)
    return self._dynamic or any(layer.dynamic for layer in self.layers)

  def _track_layers(self, layers):
    weight_layer_index = 0
    for layer_index, layer in enumerate(layers):
      try:
        if layer.weights:
          self._track_trackable(
              layer, name='layer_with_weights-%d' % weight_layer_index,
              overwrite=True)
          weight_layer_index += 1
      except ValueError:
        pass

      self._track_trackable(
          layer, name='layer-%d' % layer_index, overwrite=True)

  def __setattr__(self, name, value):
    if not getattr(self, '_self_setattr_tracking', True):
      super(Network, self).__setattr__(name, value)
      return

    if all(
        isinstance(v, (base_layer.Layer,
                       data_structures.TrackableDataStructure)) or
        trackable_layer_utils.has_weights(v) for v in nest.flatten(value)):
      try:
        self._is_graph_network
      except AttributeError:
        raise RuntimeError('It looks like you are subclassing `Model` and you '
                           'forgot to call `super(YourClass, self).__init__()`.'
                           ' Always start with this line.')

    super(Network, self).__setattr__(name, value)

    from tensorflow.python.keras import metrics as metrics_module  
    if isinstance(value, metrics_module.Metric):
      self._metrics.append(value)

  @property
  def stateful(self):
    return any((hasattr(layer, 'stateful') and layer.stateful)
               for layer in self.layers)

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
  def weights(self):
    self._assert_weights_created()
    weights = []
    for layer in self._layers:
      weights += layer.weights
    weights += (self._trainable_weights + self._non_trainable_weights)
    return weights

  @property
  @tracking.cached_per_instance
  def _should_compute_mask(self):
    return self._is_graph_network and super(Network, self)._should_compute_mask

  def compute_mask(self, inputs, mask):
    if not self._is_graph_network:
      return None

    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: t._keras_mask, output_tensors)

  @property
  def layers(self):
    return trackable_layer_utils.filter_empty_layer_containers(
        self._layers)

  def get_layer(self, name=None, index=None):
    if index is not None:
      if len(self.layers) <= index:
        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                         ' but model only has ' + str(len(self.layers)) +
                         ' layers.')
      else:
        return self.layers[index]
    else:
      if not name:
        raise ValueError('Provide either a layer name or layer index.')
    for layer in self.layers:
      if layer.name == name:
        return layer
    raise ValueError('No such layer: ' + name)

  @property
  def trainable_weights(self):
    self._assert_weights_created()
    return trackable_layer_utils.gather_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._trainable_weights)

  @property
  def non_trainable_weights(self):
    self._assert_weights_created()
    return trackable_layer_utils.gather_non_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._non_trainable_weights + self._trainable_weights)

  @property
  def input_spec(self):
    if not self._is_graph_network:
      return None

    specs = []
    for layer in self._input_layers:
      if layer.input_spec is None:
        specs.append(None)
      else:
        if not isinstance(layer.input_spec, list):
          raise TypeError('Layer ' + layer.name +
                          ' has an input_spec attribute that '
                          'is not a list. We expect a list. '
                          'Found input_spec = ' + str(layer.input_spec))
        specs += layer.input_spec
    if len(specs) == 1:
      return specs[0]
    return specs

  @base_layer_utils.default
  def build(self, input_shape):
    if self._is_graph_network:
      self.built = True
      return

    if input_shape is None:
      raise ValueError('Input shape must be defined when calling build on a '
                       'model subclass network.')
    valid_types = (tuple, list, tensor_shape.TensorShape)
    if not isinstance(input_shape, valid_types):
      raise ValueError('Specified input shape is not one of the valid types. '
                       'Please specify a batch input shape of type tuple or '
                       'list of input shapes. User provided '
                       'input type: {}'.format(type(input_shape)))

    if input_shape and not self.inputs:
      if context.executing_eagerly():
        graph = func_graph.FuncGraph('build_graph')
      else:
        graph = backend.get_graph()
      with graph.as_default():
        if isinstance(input_shape, list):
          x = [base_layer_utils.generate_placeholders_from_shape(shape)
               for shape in input_shape]
        else:
          x = base_layer_utils.generate_placeholders_from_shape(input_shape)

        kwargs = {}
        call_signature = tf_inspect.getfullargspec(self.call)
        call_args = call_signature.args
        if len(call_args) > 2:
          if call_signature.defaults:
            call_args = call_args[2:-len(call_signature.defaults)]
          else:
            call_args = call_args[2:]
          for arg in call_args:
            if arg == 'training':
              kwargs['training'] = False
            else:
              raise ValueError(
                  'Currently, you cannot build your model if it has '
                  'positional or keyword arguments that are not '
                  'inputs to the model, but are required for its '
                  '`call` method. Instead, in order to instantiate '
                  'and build your model, `call` your model on real '
                  'tensor data with all expected call arguments.')
        elif len(call_args) < 2:
          raise ValueError('You can only call `build` on a model if its `call` '
                           'method accepts an `inputs` argument.')
        try:
          self.call(x, **kwargs)
        except (errors.InvalidArgumentError, TypeError):
          raise ValueError('You cannot build your model by calling `build` '
                           'if your layers do not support float type inputs. '
                           'Instead, in order to instantiate and build your '
                           'model, `call` your model on real tensor data (of '
                           'the correct dtype).')
    if self._layers:
      self._track_layers(self._layers)
    self.built = True

  def call(self, inputs, training=None, mask=None):
    if not self._is_graph_network:
      raise NotImplementedError('When subclassing the `Model` class, you should'
                                ' implement a `call` method.')

    return self._run_internal_graph(inputs, training=training, mask=mask)

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
      return super(Network, self).compute_output_shape(input_shape)

    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    cache_key = generic_utils.object_list_uid(input_shape)
    if cache_key in self._output_shape_cache:
      return self._output_shape_cache[cache_key]

    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    if len(depth_keys) > 1:
      for depth in depth_keys:
        nodes = self._nodes_by_depth[depth]
        for node in nodes:
          layer = node.outbound_layer
          if layer in self._input_layers:
            continue
          layer_input_shapes = []
          for inbound_layer, node_id, tensor_id, _ in node.iterate_inbound():
            input_layer_key = inbound_layer.name + '_%s_%s' % (node_id,
                                                               tensor_id)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(node.inbound_layers,
                                                     layer_input_shapes)
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)

          node_index = layer._inbound_nodes.index(node)  
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape

      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      self._output_shape_cache[cache_key] = output_shapes

    return output_shapes

  def _run_internal_graph(self, inputs, training=None, mask=None):
    inputs = nest.flatten(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = nest.flatten(mask)

    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask

    tensor_dict = {}

    for x, y in zip(self.inputs, inputs):
      tensor_dict[str(id(x))] = y

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]
      for node in nodes:
        layer = node.outbound_layer

        if all(
            str(id(tensor)) in tensor_dict
            for tensor in nest.flatten(node.input_tensors)):

          computed_tensors = nest.map_structure(
              lambda t: tensor_dict[str(id(t))], node.input_tensors)

          kwargs = copy.copy(node.arguments) if node.arguments else {}
          argspec = self._layer_call_argspecs[layer].args
          if 'training' in argspec:
            kwargs.setdefault('training', training)
            if (type(kwargs['training']) is ops.Tensor and  
                any([kwargs['training'] is x
                     for x in backend._GRAPH_LEARNING_PHASES.values()])):
              kwargs['training'] = training  

          def _map_tensor_if_from_keras_layer(t):
            if isinstance(t, ops.Tensor) and hasattr(t, '_keras_history'):
              t_id = str(id(t))
              return tensor_dict[t_id]
            return t

          kwargs = nest.map_structure(_map_tensor_if_from_keras_layer, kwargs)

          output_tensors = layer(computed_tensors, **kwargs)

          for x, y in zip(
              nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
            tensor_dict[str(id(x))] = y

    output_tensors = []
    output_shapes = []
    for x in self.outputs:
      assert str(id(x)) in tensor_dict, 'Could not compute output ' + str(x)
      tensor = tensor_dict[str(id(x))]
      output_shapes.append(x.shape)
      output_tensors.append(tensor)

    if output_shapes is not None:
      input_shapes = [x.shape for x in inputs]
      cache_key = generic_utils.object_list_uid(input_shapes)
      self._output_shape_cache[cache_key] = nest.pack_sequence_as(
          self._nested_outputs, output_shapes)

    output_tensors = nest.pack_sequence_as(self._nested_outputs, output_tensors)
    return output_tensors

  def get_config(self):
    if not self._is_graph_network:
      raise NotImplementedError

    config = {
        'name': self.name,
    }
    node_conversion_map = {}
    for layer in self.layers:
      kept_nodes = 1 if _should_skip_first_node(layer) else 0
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in self._network_nodes:
          node_conversion_map[node_key] = kept_nodes
          kept_nodes += 1
    layer_configs = []
    for layer in self.layers:  
      layer_class_name = layer.__class__.__name__
      layer_config = layer.get_config()

      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in self._network_nodes:
          if node.arguments:
            kwargs = _serialize_tensors(node.arguments)
            try:
              json.dumps(kwargs)
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
            for inbound_layer, node_id, tensor_id, _ in node.iterate_inbound():
              node_key = _make_node_key(inbound_layer.name, node_id)
              new_node_index = node_conversion_map.get(node_key, 0)
              node_data.append(
                  tf_utils.ListWrapper(
                      [inbound_layer.name, new_node_index, tensor_id, kwargs]))
            node_data = nest.pack_sequence_as(node.input_tensors, node_data)
            if not nest.is_sequence(node_data):
              node_data = [node_data]
            node_data = tf_utils.convert_inner_node_data(node_data)
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
      node_key = _make_node_key(layer.name, node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_inputs.append(
          tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
    model_inputs = nest.pack_sequence_as(self._nested_inputs, model_inputs)
    if not nest.is_sequence(model_inputs):
      model_inputs = [model_inputs]
    model_inputs = tf_utils.convert_inner_node_data(model_inputs)
    config['input_layers'] = model_inputs

    model_outputs = []
    for i in range(len(self._output_layers)):
      layer, node_index, tensor_index = self._output_coordinates[i]
      node_key = _make_node_key(layer.name, node_index)
      if node_key not in self._network_nodes:
        continue
      new_node_index = node_conversion_map[node_key]
      model_outputs.append(
          tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
    model_outputs = nest.pack_sequence_as(self._nested_outputs, model_outputs)
    if not nest.is_sequence(model_outputs):
      model_outputs = [model_outputs]
    model_outputs = tf_utils.convert_inner_node_data(model_outputs)
    config['output_layers'] = model_outputs
    return copy.deepcopy(config)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    created_layers = collections.OrderedDict()

    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
      if layer not in unprocessed_nodes:
        unprocessed_nodes[layer] = [node_data]
      else:
        unprocessed_nodes[layer].append(node_data)

    def process_node(layer, node_data):
      input_tensors = []
      for input_data in nest.flatten(node_data):
        input_data = input_data.as_list()
        inbound_layer_name = input_data[0]
        inbound_node_index = input_data[1]
        inbound_tensor_index = input_data[2]
        if len(input_data) == 3:
          kwargs = {}
        elif len(input_data) == 4:
          kwargs = input_data[3]
          kwargs = _deserialize_keras_tensors(kwargs, created_layers)
        else:
          raise ValueError('Improperly formatted model config.')

        inbound_layer = created_layers[inbound_layer_name]
        if len(inbound_layer._inbound_nodes) <= inbound_node_index:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(
            nest.flatten(inbound_node.output_tensors)[inbound_tensor_index])
      input_tensors = nest.pack_sequence_as(node_data, input_tensors)
      if input_tensors is not None:
        flat_input_tensors = nest.flatten(input_tensors)
        if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
          input_tensors = flat_input_tensors[0]
        layer(input_tensors, **kwargs)

    def process_layer(layer_data):
      layer_name = layer_data['name']

      from tensorflow.python.keras.layers import deserialize as deserialize_layer  

      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

      inbound_nodes_data = layer_data['inbound_nodes']
      inbound_nodes_data = tf_utils.convert_inner_node_data(
          inbound_nodes_data, wrap=True)
      for node_data in inbound_nodes_data:
        add_unprocessed_node(layer, node_data)

    for layer_data in config['layers']:
      process_layer(layer_data)
    while unprocessed_nodes:
      for layer_data in config['layers']:
        layer = created_layers[layer_data['name']]
        if layer in unprocessed_nodes:
          for node_data in unprocessed_nodes.pop(layer):
            process_node(layer, node_data)

    name = config.get('name')
    input_tensors = []
    output_tensors = []

    input_layers = tf_utils.convert_inner_node_data(
        config['input_layers'], wrap=True)
    for layer_data in nest.flatten(input_layers):
      layer_name, node_index, tensor_index = layer_data.as_list()
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

    output_layers = tf_utils.convert_inner_node_data(
        config['output_layers'], wrap=True)
    for layer_data in nest.flatten(output_layers):
      layer_name, node_index, tensor_index = layer_data.as_list()
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

    input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
    output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
    model = cls(inputs=input_tensors, outputs=output_tensors, name=name)

    ancillary_layers = [
        layer for layer in created_layers.values() if layer not in model.layers
    ]
    if ancillary_layers:
      relevant_nodes = nest.flatten([
          layer.inbound_nodes[1:]
          if _should_skip_first_node(layer) else layer.inbound_nodes
          for layer in created_layers.values()
      ])
      model._insert_layers(ancillary_layers, relevant_nodes)
    return model

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None):
    saving.save_model(self, filepath, overwrite, include_optimizer, save_format,
                      signatures)

  def save_weights(self, filepath, overwrite=True, save_format=None):
    self._assert_weights_created()
    filepath_is_h5 = _is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        saving.save_weights_to_hdf5_group(f, self.layers)
    else:
      if context.executing_eagerly():
        session = None
      else:
        session = backend.get_session()
      optimizer = getattr(self, 'optimizer', None)
      if (optimizer
          and not isinstance(optimizer, trackable.Trackable)):
        logging.warning(
            ('This model was compiled with a Keras optimizer (%s) but is being '
             'saved in TensorFlow format with `save_weights`. The model\'s '
             'weights will be saved, but unlike with TensorFlow optimizers in '
             'the TensorFlow format the optimizer\'s state will not be '
             'saved.\n\nConsider using a TensorFlow optimizer from `tf.train`.')
            % (optimizer,))
      self._trackable_saver.save(filepath, session=session)
      checkpoint_management.update_checkpoint_state_internal(
          save_dir=os.path.dirname(filepath),
          model_checkpoint_path=filepath,
          save_relative_paths=True,
          all_model_checkpoint_paths=[filepath])

  def load_weights(self, filepath, by_name=False):
    if _is_hdf5_filepath(filepath):
      save_format = 'h5'
    else:
      try:
        pywrap_tensorflow.NewCheckpointReader(filepath)
        save_format = 'tf'
      except errors_impl.DataLossError:
        save_format = 'h5'
    if save_format == 'tf':
      status = self._trackable_saver.restore(filepath)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        trackable_utils.streaming_restore(status=status, session=session)
      status.assert_nontrivial_match()
      return status
    if h5py is None:
      raise ImportError(
          '`load_weights` requires h5py when loading weights from HDF5.')
    if self._is_graph_network and not self.built:
      raise NotImplementedError(
          'Unable to load weights saved in HDF5 format into a subclassed '
          'Model which has not created its variables yet. Call the Model '
          'first, then load the weights.')
    self._assert_weights_created()
    with h5py.File(filepath, 'r') as f:
      if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
      if by_name:
        saving.load_weights_from_hdf5_group_by_name(f, self.layers)
      else:
        saving.load_weights_from_hdf5_group(f, self.layers)

  def _updated_config(self):
    from tensorflow.python.keras import __version__ as keras_version  

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': backend.backend()
    }
    return model_config

  def to_json(self, **kwargs):
    model_config = self._updated_config()
    return json.dumps(
        model_config, default=serialization.get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    if yaml is None:
      raise ImportError(
          'Requires yaml module installed (`pip install pyyaml`).')
    return yaml.dump(self._updated_config(), **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    if not self.built:
      raise ValueError('This model has not yet been built. '
                       'Build the model first by calling `build()` or calling '
                       '`fit()` with some data, or specify '
                       'an `input_shape` argument in the first layer(s) for '
                       'automatic build.')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

  def _validate_graph_inputs_and_outputs(self):
    if len(object_identity.ObjectIdentitySet(self.inputs)) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))

    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

  def _insert_layers(self, layers, relevant_nodes=None):
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

    def _get_min_depth(node):
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound(include_arguments=True):
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          return None
      return min_depth - 1

    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')

      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
      if depth is None:  
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.outbound_layer.name,
                                node.outbound_layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)

    layer_set = set(self._layers)
    for layer in layers:
      if layer not in layer_set:
        self._layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        layer_set.add(layer)

  def _assert_weights_created(self):
    if self.dynamic:
      return
    if (not self._is_graph_network and
        'build' in self.__class__.__dict__ and
        not self.built):
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  @property
  def _object_identifier(self):
    return '_tf_keras_network'

  def _graph_network_add_loss(self, symbolic_loss):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    add_loss_layer = base_layer.AddLoss(unconditional=False)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)

  def _graph_network_add_metric(self, value, aggregation, name):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(aggregation, name)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)

def _is_hdf5_filepath(filepath):
  return (filepath.endswith('.h5') or filepath.endswith('.keras') or
          filepath.endswith('.hdf5'))

def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)

def _map_graph_network(inputs, outputs):
  network_nodes = set()  
  nodes_depths = {}  
  layers_depths = {}  
  layer_indices = {}  
  nodes_in_decreasing_depth = []

  def build_map(tensor,
                finished_nodes,
                nodes_in_progress,
                layer,
                node_index,
                tensor_index):
    node = layer._inbound_nodes[node_index]  

    if node in nodes_in_progress:
      raise ValueError('The tensor ' + str(tensor) + ' at layer "' +
                       layer.name + '" is part of a cycle.')

    if node in finished_nodes:
      return

    node_key = _make_node_key(layer.name, node_index)
    network_nodes.add(node_key)

    if layer not in layer_indices:
      layer_indices[layer] = len(layer_indices)

    nodes_in_progress.add(node)

    for layer, node_index, tensor_index, tensor in node.iterate_inbound(
        include_arguments=True):
      build_map(tensor, finished_nodes, nodes_in_progress, layer, node_index,
                tensor_index)

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)

  finished_nodes = set()
  nodes_in_progress = set()
  for x in outputs:
    layer, node_index, tensor_index = x._keras_history  
    build_map(x, finished_nodes, nodes_in_progress,
              layer=layer,
              node_index=node_index,
              tensor_index=tensor_index)

  for node in reversed(nodes_in_decreasing_depth):
    depth = nodes_depths.setdefault(node, 0)

    previous_depth = layers_depths.get(node.outbound_layer, 0)
    depth = max(depth, previous_depth)
    layers_depths[node.outbound_layer] = depth
    nodes_depths[node] = depth

    for node_dep in node._get_all_node_dependencies():
      previous_depth = nodes_depths.get(node_dep, 0)
      nodes_depths[node_dep] = max(depth + 1, previous_depth)

  for input_t in inputs:
    input_layer = input_t._keras_history[0]
    if input_layer not in layers_depths:
      layers_depths[input_layer] = 0
      layer_indices[input_layer] = -1
      nodes_depths[input_layer._inbound_nodes[0]] = 0
      network_nodes.add(_make_node_key(input_layer.name, 0))

  nodes_by_depth = collections.defaultdict(list)
  for node, depth in nodes_depths.items():
    nodes_by_depth[depth].append(node)

  layers_by_depth = collections.defaultdict(list)
  for layer, depth in layers_depths.items():
    layers_by_depth[depth].append(layer)

  depth_keys = list(layers_by_depth.keys())
  depth_keys.sort(reverse=True)

  layers = []
  for depth in depth_keys:
    layers_for_depth = layers_by_depth[depth]
    layers_for_depth.sort(key=lambda x: layer_indices[x])
    layers.extend(layers_for_depth)

  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  computable_tensors = object_identity.ObjectIdentitySet()
  for x in inputs:
    computable_tensors.add(x)

  layers_with_complete_input = []  
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      layer = node.outbound_layer
      if layer:
        for x in nest.flatten(node.input_tensors):
          if x not in computable_tensors:
            raise ValueError('Graph disconnected: '
                             'cannot obtain value for tensor ' + str(x) +
                             ' at layer "' + layer.name + '". '
                             'The following previous layers '
                             'were accessed without issue: ' +
                             str(layers_with_complete_input))
        for x in nest.flatten(node.output_tensors):
          computable_tensors.add(x)
        layers_with_complete_input.append(layer.name)

  all_names = [layer.name for layer in layers]
  for name in all_names:
    if all_names.count(name) != 1:
      raise ValueError('The name "' + name + '" is used ' +
                       str(all_names.count(name)) + ' times in the model. '
                       'All layer names should be unique.')
  return network_nodes, nodes_by_depth, layers, layers_by_depth

def _map_subgraph_network(inputs, outputs):
  base_layer_utils.create_keras_history(outputs)
  _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
  return nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers

def _should_skip_first_node(layer):
  return issubclass(layer.__class__, Network) and layer._is_graph_network

def _serialize_tensors(kwargs):

  def _serialize_keras_tensor(t):
    if hasattr(t, '_keras_history'):
      kh = t._keras_history
      return [kh.layer.name, kh.node_index, kh.tensor_index]

    if isinstance(t, np.ndarray):
      return t.tolist()

    if isinstance(t, ops.Tensor):
      return backend.get_value(t).tolist()

    return t

  return nest.map_structure(_serialize_keras_tensor, kwargs)

def _deserialize_keras_tensors(kwargs, layer_map):

  def _deserialize_keras_tensor(t):
    if isinstance(t, tf_utils.ListWrapper):
      t = t.as_list()
      layer_name = t[0]
      node_index = t[1]
      tensor_index = t[2]

      layer = layer_map[layer_name]
      node = layer._inbound_nodes[node_index]
      return nest.flatten(node.output_tensors)[tensor_index]
    return t

  kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
  return nest.map_structure(_deserialize_keras_tensor, kwargs)
EOF
