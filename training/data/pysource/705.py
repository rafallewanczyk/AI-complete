

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import types

from tensorflow.python.eager import context
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking.tracking import delete_tracking
from tensorflow.python.util import compat
from tensorflow.python.util import nest

models_lib = LazyLoader("models_lib", globals(),
                        "tensorflow.python.keras.models")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
layers_module = LazyLoader(
    "layers_module", globals(),
    "tensorflow.python.keras.layers")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")
functional_lib = LazyLoader(
    "functional_lib", globals(),
    "tensorflow.python.keras.engine.functional")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
training_lib_v1 = LazyLoader(
    "training_lib_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
recurrent = LazyLoader(
    "recurrent", globals(),
    "tensorflow.python.keras.layers.recurrent")

PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(constants.KERAS_ATTR)

KERAS_OBJECT_IDENTIFIERS = (
    '_tf_keras_layer', '_tf_keras_input_layer', '_tf_keras_network',
    '_tf_keras_model', '_tf_keras_sequential', '_tf_keras_metric',
    '_tf_keras_rnn_layer')

def load(path, compile=True, options=None):  

  model = tf_load.load_internal(
      path, options=options, loader_cls=KerasObjectLoader)

  if isinstance(model, training_lib.Model) and compile:
    training_config = model._serialized_attributes['metadata'].get(
        'training_config', None)
    if training_config is not None:
      model.compile(**saving_utils.compile_args_from_training_config(
          training_config))
      saving_utils.try_build_compiled_arguments(model)
    else:
      logging.warning('No training configuration found in save file, so the '
                      'model was *not* compiled. Compile it manually.')

  if not context.executing_eagerly():
    sess = backend.get_session()  
    sess.run(ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS))

  return model

def _is_graph_network(layer):
  if isinstance(layer, RevivedNetwork):
    return False
  elif isinstance(layer, functional_lib.Functional):
    return (layer._is_graph_network or
            isinstance(layer, models_lib.Sequential))
  return False

class KerasObjectLoader(tf_load.Loader):

  def __init__(self, *args, **kwargs):
    self._nodes_recreated_from_config = {}
    self._traversed_nodes_from_config = []

    self.model_layer_dependencies = {}
    self._models_to_reconstruct = []

    super(KerasObjectLoader, self).__init__(*args, **kwargs)

    for node in self._nodes:
      if not isinstance(node, base_layer.Layer):
        continue
      for name in PUBLIC_ATTRIBUTES:
        delete_tracking(node, name)

  def _load_all(self):
    self._layer_nodes = self._load_layers()

    super(KerasObjectLoader, self)._load_all()

    self._finalize_objects()

  @property
  def _expect_partial_checkpoint(self):
    return True

  def _recreate(self, proto, node_id):
    if node_id in self._layer_nodes:
      return self._layer_nodes[node_id]

    if node_id in self._nodes_recreated_from_config:
      obj, setter = self._nodes_recreated_from_config[node_id]

      if proto.WhichOneof('kind') == 'variable' and proto.variable.name:
        obj._handle_name = proto.variable.name + ':0'  
    else:
      obj, setter = super(KerasObjectLoader, self)._recreate(proto, node_id)
    return obj, setter

  def _add_children_recreated_from_config(self, obj, proto, node_id):
    if node_id in self._traversed_nodes_from_config:
      return
    self._traversed_nodes_from_config.append(node_id)
    obj._maybe_initialize_trackable()
    if isinstance(obj, base_layer.Layer) and not obj.built:
      metadata = json_utils.decode(proto.user_object.metadata)
      self._try_build_layer(obj, node_id, metadata.get('build_input_shape'))

    children = []
    for reference in proto.children:
      obj_child = obj._lookup_dependency(reference.local_name)
      children.append((obj_child, reference.node_id))

    metric_list_node_id = self._search_for_child_node(
        node_id, [constants.KERAS_ATTR, 'layer_metrics'], raise_error=False)
    if metric_list_node_id is not None and hasattr(obj, '_metrics'):
      obj_metrics = {m.name: m for m in obj._metrics}
      for reference in self._proto.nodes[metric_list_node_id].children:
        metric = obj_metrics.get(reference.local_name)
        if metric is not None:
          children.append((metric, reference.node_id))

    for (obj_child, child_id) in children:
      child_proto = self._proto.nodes[child_id]

      if not isinstance(obj_child, trackable.Trackable):
        continue
      if (child_proto.user_object.identifier in
          revived_types.registered_identifiers()):
        setter = revived_types.get_setter(child_proto.user_object)
      elif obj_child._object_identifier in KERAS_OBJECT_IDENTIFIERS:
        setter = _revive_setter
      else:
        setter = setattr

      if (child_id in self._nodes_recreated_from_config and
          self._nodes_recreated_from_config[child_id][0] is not obj_child):
        logging.warn('Looks like there is an object (perhaps variable or layer)'
                     ' that is shared between different layers/models. This '
                     'may cause issues when restoring the variable values.'
                     'Object: {}'.format(obj_child))
      self._nodes_recreated_from_config[child_id] = (
          obj_child, self._config_node_setter(setter))
      self._add_children_recreated_from_config(
          obj_child, child_proto, child_id)

  def _load_layers(self):
    layers = {}

    metric_list = []
    for node_id, proto in enumerate(self._proto.nodes):
      if (proto.WhichOneof('kind') != 'user_object' or
          proto.user_object.identifier not in KERAS_OBJECT_IDENTIFIERS):
        continue
      if proto.user_object.identifier == '_tf_keras_metric':
        metric_list.append((node_id, proto))
        continue

      layers[node_id] = self._load_layer(proto.user_object, node_id)

    for node_id, proto in metric_list:
      layers[node_id] = self._load_layer(proto.user_object, node_id)
    return layers

  def _load_layer(self, proto, node_id):
    metadata = json_utils.decode(proto.metadata)

    if node_id in self._nodes_recreated_from_config:
      node, setter = self._nodes_recreated_from_config[node_id]

      _maybe_add_serialized_attributes(node, metadata)

      config = metadata.get('config')
      if _is_graph_network(node) and generic_utils.validate_config(config):
        self.model_layer_dependencies[node_id] = (
            node, self._get_child_layer_node_ids(node_id, node.name))
      return node, setter

    obj, setter = self._revive_from_config(proto.identifier, metadata, node_id)
    if obj is None:
      obj, setter = revive_custom_object(proto.identifier, metadata)

    _maybe_add_serialized_attributes(obj, metadata)
    return obj, setter

  def _revive_from_config(self, identifier, metadata, node_id):
    if identifier == '_tf_keras_metric':
      obj = self._revive_metric_from_config(metadata, node_id)
    else:
      obj = (
          self._revive_graph_network(metadata, node_id) or
          self._revive_layer_from_config(metadata, node_id))

    if obj is None:
      return None, None

    setter = self._config_node_setter(_revive_setter)
    self._nodes_recreated_from_config[node_id] = obj, setter
    self._add_children_recreated_from_config(
        obj, self._proto.nodes[node_id], node_id)
    return obj, setter

  def _revive_graph_network(self, metadata, node_id):
    class_name = compat.as_str(metadata['class_name'])
    config = metadata.get('config')

    model_is_functional_or_sequential = (
        metadata.get('is_graph_network', False) or
        metadata['class_name'] == 'Sequential' or
        metadata['class_name'] == 'Functional')
    if not (generic_utils.validate_config(config) and
            model_is_functional_or_sequential
           ) or generic_utils.get_registered_object(class_name) is not None:
      return None

    if class_name == 'Sequential':
      model = models_lib.Sequential(name=config['name'])
    else:
      model = models_lib.Functional(
          inputs=[], outputs=[], name=config['name'])

    layers = self._get_child_layer_node_ids(node_id, model.name)
    self.model_layer_dependencies[node_id] = (model, layers)

    return model

  def _revive_layer_from_config(self, metadata, node_id):
    class_name = metadata.get('class_name')
    config = metadata.get('config')
    must_restore_from_config = metadata.get('must_restore_from_config')
    if not generic_utils.validate_config(config):
      return None

    try:
      obj = layers_module.deserialize(
          generic_utils.serialize_keras_class_and_config(class_name, config))
    except ValueError:
      if must_restore_from_config:
        raise RuntimeError(
            'Unable to restore a layer of class {cls}. Layers of '
            'class {cls} require that the class be provided to '
            'the model loading code, either by registering the '
            'class using @keras.utils.register_keras_serializable '
            'on the class def and including that file in your '
            'program, or by passing the class in a '
            'keras.utils.CustomObjectScope that wraps this load '
            'call.'.format(cls=class_name))
      else:
        return None

    obj._name = metadata['name']
    if metadata.get('trainable') is not None:
      obj.trainable = metadata['trainable']
    if metadata.get('dtype') is not None:
      obj._set_dtype_policy(metadata['dtype'])
    if metadata.get('stateful') is not None:
      obj.stateful = metadata['stateful']

    build_input_shape = metadata.get('build_input_shape')
    built = self._try_build_layer(obj, node_id, build_input_shape)

    if not built:
      return None

    return obj

  def _revive_metric_from_config(self, metadata, node_id):
    class_name = compat.as_str(metadata['class_name'])
    config = metadata.get('config')

    if not generic_utils.validate_config(config):
      return None

    try:
      obj = metrics.deserialize(
          generic_utils.serialize_keras_class_and_config(class_name, config))
    except ValueError:
      return None

    build_input_shape = metadata.get('build_input_shape')
    if build_input_shape is not None and hasattr(obj, '_build'):
      obj._build(build_input_shape)  

    return obj

  def _try_build_layer(self, obj, node_id, build_input_shape):
    if obj.built or hasattr(obj.build, '_is_default'):
      obj.built = True
      return True

    if build_input_shape is None:
      build_input_shape = self._infer_inputs(node_id, convert_to_shapes=True)

    if build_input_shape is not None:
      obj.build(build_input_shape)
      base_layer.Layer.build(obj, build_input_shape)
      return True

    return False

  def _load_edges(self):
    for node_id, proto in enumerate(self._proto.nodes):
      if node_id not in self.model_layer_dependencies:
        self._add_object_graph_edges(proto, node_id)

  def _finalize_objects(self):
    layers_revived_from_config = []
    layers_revived_from_saved_model = []
    for node_id, node in enumerate(self._nodes):
      if (not isinstance(node, base_layer.Layer) or
          node_id in self.model_layer_dependencies):
        continue

      self._unblock_model_reconstruction(node_id, node)

      if isinstance(node, input_layer.InputLayer):
        continue
      elif isinstance(node, metrics.Metric):
        continue

      if node_id in self._nodes_recreated_from_config:
        layers_revived_from_config.append(node)
      else:
        layers_revived_from_saved_model.append(node)

    _finalize_saved_model_layers(layers_revived_from_saved_model)
    _finalize_config_layers(layers_revived_from_config)

    self._reconstruct_all_models()

  def _unblock_model_reconstruction(self, layer_id, layer):
    for model_id, v in self.model_layer_dependencies.items():
      _, layers = v
      if layer_id not in layers:
        continue
      layers[layers.index(layer_id)] = layer
      if all(isinstance(x, base_layer.Layer) for x in layers):
        self._models_to_reconstruct.append(model_id)

  def _reconstruct_all_models(self):
    all_initialized_models = set()
    while self._models_to_reconstruct:
      model_id = self._models_to_reconstruct.pop(0)
      all_initialized_models.add(model_id)
      model, layers = self.model_layer_dependencies[model_id]
      self._reconstruct_model(model_id, model, layers)
      self._add_object_graph_edges(self._proto.nodes[model_id], model_id)
      _finalize_config_layers([model])

    if all_initialized_models != set(self.model_layer_dependencies.keys()):
      uninitialized_model_ids = (
          set(self.model_layer_dependencies.keys()) - all_initialized_models)
      uninitialized_model_names = [
          self.model_layer_dependencies[model_id][0].name
          for model_id in uninitialized_model_ids]
      raise ValueError('Error when loading from SavedModel -- the following '
                       'models could not be initialized: {}'
                       .format(uninitialized_model_names))

  def _reconstruct_model(self, model_id, model, layers):
    config = json_utils.decode(
        self._proto.nodes[model_id].user_object.metadata)['config']
    if isinstance(model, models_lib.Sequential):
      if not isinstance(layers[0], input_layer.InputLayer):
        if config['layers'][0]['class_name'] == 'InputLayer':
          layers.insert(0, input_layer.InputLayer.from_config(
              config['layers'][0]['config']))
        elif 'batch_input_shape' in config['layers'][0]['config']:
          batch_input_shape = config['layers'][0]['config']['batch_input_shape']
          layers.insert(0, input_layer.InputLayer(
              input_shape=batch_input_shape[1:],
              batch_size=batch_input_shape[0],
              dtype=layers[0].dtype,
              name=layers[0].name + '_input'))
      model.__init__(layers, name=config['name'])
      if not model.inputs:
        first_layer = self._get_child_layer_node_ids(model_id, model.name)[0]
        input_specs = self._infer_inputs(first_layer)
        input_shapes = self._infer_inputs(first_layer, convert_to_shapes=True)
        model._set_inputs(input_specs)  
        if not model.built and not isinstance(input_specs, dict):
          model.build(input_shapes)
    else:
      (inputs, outputs,
       created_layers) = functional_lib.reconstruct_from_config(
           config, created_layers={layer.name: layer for layer in layers})
      model.__init__(inputs, outputs, name=config['name'])
      functional_lib.connect_ancillary_layers(model, created_layers)

    _set_network_attributes_from_metadata(model)

    self._unblock_model_reconstruction(model_id, model)

  def _get_child_layer_node_ids(self, node_id, name):
    layer_list = self._search_for_child_node(
        node_id, [constants.KERAS_ATTR, 'layers'], name)
    return [node.node_id for node in self._proto.nodes[layer_list].children]

  def _search_for_child_node(
      self, parent_id, path_to_child, debugging_name=None, raise_error=True):
    if not path_to_child:
      return parent_id

    for child in self._proto.nodes[parent_id].children:
      if child.local_name == path_to_child[0]:
        return self._search_for_child_node(child.node_id, path_to_child[1:],
                                           debugging_name, raise_error)

    if raise_error:
      raise ValueError(
          'Error when loading {}: could not find attribute {}.\n'
          'Most likely this object was serialized incorrectly.'
          .format(debugging_name or path_to_child[0], path_to_child[0]))
    else:
      return None

  def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
    coder = nested_structure_coder.StructureCoder()
    call_fn_id = self._search_for_child_node(
        layer_node_id, ['call_and_return_all_conditional_losses'], None,
        raise_error=False)
    if call_fn_id is None:
      return None

    concrete_functions = (
        self._proto.nodes[call_fn_id].function.concrete_functions)
    if not concrete_functions:
      return None
    call_fn_name = concrete_functions[0]
    call_fn_proto = self._proto.concrete_functions[call_fn_name]
    structured_input_signature = coder.decode_proto(
        call_fn_proto.canonicalized_input_signature)
    inputs = structured_input_signature[0][0]
    if convert_to_shapes:
      return nest.map_structure(lambda spec: spec.shape, inputs)
    else:
      return inputs

  def _config_node_setter(self, setter):
    def setattr_wrapper(obj, name, value):
      if obj._lookup_dependency(name) is None:  
        setter(obj, name, value)
    return setattr_wrapper

def _finalize_saved_model_layers(layers):
  for layer in layers:
    layer.built = True
    if hasattr(_get_keras_attr(layer), 'call_and_return_conditional_losses'):
      layer.call = utils.use_wrapped_call(
          layer, _get_keras_attr(layer).call_and_return_conditional_losses,
          return_method=True)
      layer._init_call_fn_args()
    else:
      layer.call = types.MethodType(
          _unable_to_call_layer_due_to_serialization_issue, layer)

  for layer in layers:
    if isinstance(layer, RevivedNetwork):
      _set_network_attributes_from_metadata(layer)

      if hasattr(_get_keras_attr(layer), 'call_and_return_conditional_losses'):
        call_fn = _get_keras_attr(layer).call_and_return_conditional_losses
        if call_fn.input_signature is None:
          inputs = infer_inputs_from_restored_call_function(call_fn)
        else:
          inputs = call_fn.input_signature[0]
        layer._set_inputs(inputs)  

    _restore_layer_unconditional_losses(layer)
    _restore_layer_activation_loss(layer)

    _restore_layer_metrics(layer)

def _unable_to_call_layer_due_to_serialization_issue(
    layer, *unused_args, **unused_kwargs):

  raise ValueError(
      'Cannot call {} ({}), because the call function was not serialized to '
      'the SavedModel (due to lack information about the inputs). Please try '
      'one of the following methods to fix the serialization:'
      '\n\n(1) Implement `get_config` and `from_config` in the layer/model '
      'class, and pass the object to the `custom_objects` argument when '
      'loading the model. For more details, see: '
      'https://www.tensorflow.org/guide/keras/save_and_serialize'
      '\n\n(2) Ensure that the subclassed model or layer overwrites `call` '
      'and not `__call__`. The input shape and dtype will be automatically '
      'recorded when the object is called, and used when saving. To manually '
      'specify the input shape/dtype, decorate the call function with '
      '`@tf.function(input_signature=...)`.'.format(layer.name, layer))

def _finalize_config_layers(layers):
  for layer in layers:
    if _is_graph_network(layer):
      _restore_layer_unconditional_losses(layer)

    _restore_layer_activation_loss(layer)

    _restore_layer_metrics(layer)

    if (isinstance(layer, recurrent.RNN) and
        layer.stateful and
        hasattr(_get_keras_attr(layer), 'states')):
      layer.states = getattr(_get_keras_attr(layer), 'states', None)
      for variable in nest.flatten(layer.states):
        backend.track_variable(variable)

def _finalize_metric(metric):
  metric.update_state = types.MethodType(metrics_utils.update_state_wrapper(
      metric.keras_api.update_state), metric)
  metric.result = metric.keras_api.result

def _restore_layer_unconditional_losses(layer):
  if hasattr(_get_keras_attr(layer), 'layer_regularization_losses'):
    losses = getattr(_get_keras_attr(layer), 'layer_regularization_losses', [])
  else:
    losses = layer._serialized_attributes.get('regularization_losses', [])  
  for loss in losses:
    layer.add_loss(loss)

def _restore_layer_activation_loss(layer):
  activity_regularizer = getattr(_get_keras_attr(layer),
                                 'activity_regularizer_fn', None)
  if activity_regularizer and not layer.activity_regularizer:
    try:
      layer.activity_regularizer = activity_regularizer
    except AttributeError:
      pass

def revive_custom_object(identifier, metadata):
  if ops.executing_eagerly_outside_functions():
    model_class = training_lib.Model
  else:
    model_class = training_lib_v1.Model

  revived_classes = {
      '_tf_keras_layer': (RevivedLayer, base_layer.Layer),
      '_tf_keras_input_layer': (RevivedInputLayer, input_layer.InputLayer),
      '_tf_keras_network': (RevivedNetwork, functional_lib.Functional),
      '_tf_keras_model': (RevivedNetwork, model_class),
      '_tf_keras_sequential': (RevivedNetwork, models_lib.Sequential),
  }
  parent_classes = revived_classes.get(identifier, None)

  if parent_classes is not None:
    parent_classes = revived_classes[identifier]
    revived_cls = type(
        compat.as_str(metadata['class_name']), parent_classes, {})
    return revived_cls._init_from_metadata(metadata)  
  else:
    raise ValueError('Unable to restore custom object of type {} currently. '
                     'Please make sure that the layer implements `get_config`'
                     'and `from_config` when saving. In addition, please use '
                     'the `custom_objects` arg when calling `load_model()`.'
                     .format(identifier))

def _restore_layer_metrics(layer):
  metrics_list = getattr(_get_keras_attr(layer), 'layer_metrics', {})
  layer_metrics = {m.name: m for m in layer._metrics}  
  for name, metric in metrics_list.items():
    if name not in layer_metrics:
      layer._metrics.append(metric)  

class RevivedLayer(object):

  @classmethod
  def _init_from_metadata(cls, metadata):
    init_args = dict(
        name=metadata['name'],
        trainable=metadata['trainable'])
    if metadata.get('dtype') is not None:
      init_args['dtype'] = metadata['dtype']
    if metadata.get('batch_input_shape') is not None:
      init_args['batch_input_shape'] = metadata['batch_input_shape']

    revived_obj = cls(**init_args)

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      revived_obj._expects_training_arg = metadata['expects_training_arg']
      config = metadata.get('config')
      if generic_utils.validate_config(config):
        revived_obj._config = config
      if metadata.get('input_spec') is not None:
        revived_obj.input_spec = recursively_deserialize_keras_object(
            metadata['input_spec'],
            module_objects={'InputSpec': input_spec.InputSpec})
      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])
      if metadata.get('_is_feature_layer') is not None:
        revived_obj._is_feature_layer = metadata['_is_feature_layer']
      if metadata.get('stateful') is not None:
        revived_obj.stateful = metadata['stateful']

    return revived_obj, _revive_setter

  @property
  def keras_api(self):
    return self._serialized_attributes.get(constants.KERAS_ATTR, None)

  def get_config(self):
    if hasattr(self, '_config'):
      return self._config
    else:
      raise NotImplementedError

def _revive_setter(layer, name, value):
  if name in PUBLIC_ATTRIBUTES:
    if isinstance(value, trackable.Trackable):
      layer._track_trackable(value, name=name)
    layer._serialized_attributes[name] = value
  elif (isinstance(layer, functional_lib.Functional) and
        re.match(r'^layer(_with_weights)?-[\d+]', name) is not None):
    pass
  elif getattr(layer, name, None) is not None:
    pass
  else:
    setattr(layer, name, value)

class RevivedInputLayer(object):

  @classmethod
  def _init_from_metadata(cls, metadata):
    init_args = dict(
        name=metadata['name'],
        dtype=metadata['dtype'],
        sparse=metadata['sparse'],
        ragged=metadata['ragged'],
        batch_input_shape=metadata['batch_input_shape'])
    revived_obj = cls(**init_args)
    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      revived_obj._config = metadata['config']  

    return revived_obj, setattr

  def get_config(self):
    return self._config

def recursively_deserialize_keras_object(config, module_objects=None):
  if isinstance(config, dict):
    if 'class_name' in config:
      return generic_utils.deserialize_keras_object(
          config, module_objects=module_objects)
    else:
      return {key: recursively_deserialize_keras_object(config[key],
                                                        module_objects)
              for key in config}
  if isinstance(config, (tuple, list)):
    return [recursively_deserialize_keras_object(x, module_objects)
            for x in config]
  else:
    raise ValueError('Unable to decode config: {}'.format(config))

def infer_inputs_from_restored_call_function(fn):
  def common_spec(x, y):
    return tensor_spec.TensorSpec(defun.common_shape(x.shape, y.shape),
                                  x.dtype, x.name)
  spec = fn.concrete_functions[0].structured_input_signature[0][0]
  for concrete in fn.concrete_functions[1:]:
    spec2 = concrete.structured_input_signature[0][0]
    spec = nest.map_structure(common_spec, spec, spec2)
  return spec

class RevivedNetwork(RevivedLayer):

  @classmethod
  def _init_from_metadata(cls, metadata):
    revived_obj = cls(name=metadata['name'])

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      revived_obj._expects_training_arg = metadata['expects_training_arg']
      config = metadata.get('config')
      if generic_utils.validate_config(config):
        revived_obj._config = config

      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])

    return revived_obj, _revive_setter  

def _set_network_attributes_from_metadata(revived_obj):
  with trackable.no_automatic_dependency_tracking_scope(revived_obj):
    metadata = revived_obj._serialized_attributes['metadata']
    if metadata.get('dtype') is not None:
      revived_obj._set_dtype_policy(metadata['dtype'])
    revived_obj.trainable = metadata['trainable']

def _maybe_add_serialized_attributes(layer, metadata):
  if not hasattr(layer, '_serialized_attributes'):
    with trackable.no_automatic_dependency_tracking_scope(layer):
      layer._serialized_attributes = {'metadata': metadata}  

def _get_keras_attr(layer):
  return getattr(layer, '_serialized_attributes', {}).get(constants.KERAS_ATTR,
                                                          None)
EOF
