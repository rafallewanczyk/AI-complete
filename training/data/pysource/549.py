

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import os
import sys
import threading
import weakref

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tfdev
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.training.tracking import util as tracking_util
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import keras_export

py_all = all
py_sum = sum
py_any = any

_GRAPH = threading.local()

_CURRENT_SCRATCH_GRAPH = threading.local()

_SESSION = threading.local()

class _DummyEagerGraph(threading.local):

  class _WeakReferencableClass(object):
    pass

  def __init__(self):
    super(_DummyEagerGraph, self).__init__()
    self.key = _DummyEagerGraph._WeakReferencableClass()
    self.learning_phase_is_set = False

_DUMMY_EAGER_GRAPH = _DummyEagerGraph()

_MANUAL_VAR_INIT = False

_LOCAL_DEVICES = None

epsilon = backend_config.epsilon
floatx = backend_config.floatx
image_data_format = backend_config.image_data_format
set_epsilon = backend_config.set_epsilon
set_floatx = backend_config.set_floatx
set_image_data_format = backend_config.set_image_data_format

@keras_export('keras.backend.backend')
def backend():
  return 'tensorflow'

@keras_export('keras.backend.cast_to_floatx')
@dispatch.add_dispatch_support
def cast_to_floatx(x):
  if isinstance(x, (ops.Tensor,
                    variables_module.Variable,
                    sparse_tensor.SparseTensor)):
    return math_ops.cast(x, dtype=floatx())
  return np.asarray(x, dtype=floatx())

PER_GRAPH_OBJECT_NAME_UIDS = weakref.WeakKeyDictionary()

@keras_export('keras.backend.get_uid')
def get_uid(prefix=''):
  graph = get_graph()
  if graph not in PER_GRAPH_OBJECT_NAME_UIDS:
    PER_GRAPH_OBJECT_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_OBJECT_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]

@keras_export('keras.backend.reset_uids')
def reset_uids():

  PER_GRAPH_OBJECT_NAME_UIDS.clear()

@keras_export('keras.backend.clear_session')
def clear_session():
  global _SESSION
  global _GRAPH_LEARNING_PHASES  
  global _GRAPH_VARIABLES  
  global _GRAPH_TF_OPTIMIZERS  
  global _GRAPH
  _GRAPH.graph = None
  ops.reset_default_graph()
  reset_uids()
  _SESSION.session = None
  graph = get_graph()
  with graph.as_default():
    _DUMMY_EAGER_GRAPH.learning_phase_is_set = False
    _GRAPH_LEARNING_PHASES.clear()
    _GRAPH_LEARNING_PHASES.setdefault(graph)
    _GRAPH_VARIABLES.pop(graph, None)
    _GRAPH_TF_OPTIMIZERS.pop(graph, None)

@keras_export('keras.backend.manual_variable_initialization')
def manual_variable_initialization(value):
  global _MANUAL_VAR_INIT
  _MANUAL_VAR_INIT = value

@keras_export('keras.backend.learning_phase')
def learning_phase():
  graph = ops.get_default_graph()
  if graph is getattr(_GRAPH, 'graph', None):
    learning_phase = symbolic_learning_phase()
  else:
    with ops.init_scope():
      learning_phase = _GRAPH_LEARNING_PHASES[None]
  _mark_func_graph_as_unsaveable(graph, learning_phase)
  return learning_phase

def global_learning_phase_is_set():
  return _DUMMY_EAGER_GRAPH.learning_phase_is_set

def _mark_func_graph_as_unsaveable(graph, learning_phase):
  if graph.building_function and is_placeholder(learning_phase):
    graph.mark_as_unsaveable(
        'The keras learning phase placeholder was used inside a function. '
        'Exporting placeholders is not supported when saving out a SavedModel. '
        'Please call `tf.keras.backend.set_learning_phase(0)` in the function '
        'to set the learning phase to a constant value.')

def symbolic_learning_phase():
  graph = get_graph()
  with graph.as_default():
    return _GRAPH_LEARNING_PHASES[graph]

def _default_learning_phase():
  if context.executing_eagerly():
    return 0
  else:
    with name_scope(''):
      return array_ops.placeholder_with_default(
          False, shape=(), name='keras_learning_phase')

@deprecated('2020-10-11',
            'Simply pass a True/False value to the `training` argument '
            'of the `__call__` method of your layer or model.')
@keras_export('keras.backend.set_learning_phase')
def set_learning_phase(value):
  deprecated_internal_set_learning_phase(value)

def deprecated_internal_set_learning_phase(value):
  global _GRAPH_LEARNING_PHASES  
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  with ops.init_scope():
    if context.executing_eagerly():
      _DUMMY_EAGER_GRAPH.learning_phase_is_set = True
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = value
    _GRAPH_LEARNING_PHASES[get_graph()] = value

@deprecated('2020-10-11',
            'Simply pass a True/False value to the `training` argument '
            'of the `__call__` method of your layer or model.')
@keras_export('keras.backend.learning_phase_scope')
@tf_contextlib.contextmanager
def learning_phase_scope(value):
  with deprecated_internal_learning_phase_scope(value):
    try:
      yield
    finally:
      pass

@tf_contextlib.contextmanager
def deprecated_internal_learning_phase_scope(value):
  global _GRAPH_LEARNING_PHASES  
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')

  with ops.init_scope():
    if context.executing_eagerly():
      previous_eager_value = _GRAPH_LEARNING_PHASES.get(
          _DUMMY_EAGER_GRAPH.key, None)
    previous_graph_value = _GRAPH_LEARNING_PHASES.get(get_graph(), None)

  learning_phase_previously_set = _DUMMY_EAGER_GRAPH.learning_phase_is_set
  try:
    deprecated_internal_set_learning_phase(value)
    yield
  finally:
    if not learning_phase_previously_set:
      _DUMMY_EAGER_GRAPH.learning_phase_is_set = False
    with ops.init_scope():
      if context.executing_eagerly():
        if previous_eager_value is not None:
          _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = previous_eager_value
        elif _DUMMY_EAGER_GRAPH.key in _GRAPH_LEARNING_PHASES:
          del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key]

      graph = get_graph()
      if previous_graph_value is not None:
        _GRAPH_LEARNING_PHASES[graph] = previous_graph_value
      elif graph in _GRAPH_LEARNING_PHASES:
        del _GRAPH_LEARNING_PHASES[graph]

@tf_contextlib.contextmanager
def eager_learning_phase_scope(value):
  global _GRAPH_LEARNING_PHASES  
  assert value in {0, 1}
  assert ops.executing_eagerly_outside_functions()
  global_learning_phase_was_set = global_learning_phase_is_set()
  if global_learning_phase_was_set:
    previous_value = learning_phase()
  try:
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = value
    yield
  finally:
    if global_learning_phase_was_set:
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = previous_value
    else:
      del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key]

def _current_graph(op_input_list):
  return ops._get_graph_from_inputs(op_input_list)

def _get_session(op_input_list=()):
  global _SESSION
  default_session = ops.get_default_session()
  if default_session is not None:
    session = default_session
  else:
    if ops.inside_function():
      raise RuntimeError('Cannot get session inside Tensorflow graph function.')
    if (getattr(_SESSION, 'session', None) is None or
        _SESSION.session.graph is not _current_graph(op_input_list)):
      if distribution_strategy_context.has_strategy():
        configure_and_create_distributed_session(
            distribution_strategy_context.get_strategy())
      else:
        _SESSION.session = session_module.Session(
            config=get_default_session_config())
    session = _SESSION.session
  return session

@keras_export(v1=['keras.backend.get_session'])
def get_session(op_input_list=()):
  session = _get_session(op_input_list)
  if not _MANUAL_VAR_INIT:
    with session.graph.as_default():
      _initialize_variables(session)
  return session

tracking_util.register_session_provider(get_session)

def get_graph():
  if context.executing_eagerly():
    global _GRAPH
    if not getattr(_GRAPH, 'graph', None):
      _GRAPH.graph = func_graph.FuncGraph('keras_graph')
    return _GRAPH.graph
  else:
    return ops.get_default_graph()

@tf_contextlib.contextmanager
def _scratch_graph(graph=None):
  global _CURRENT_SCRATCH_GRAPH
  scratch_graph = getattr(_CURRENT_SCRATCH_GRAPH, 'graph', None)
  if (scratch_graph is not None and graph is not None and
      scratch_graph is not graph):
    raise ValueError('Multiple scratch graphs specified.')

  if scratch_graph:
    yield scratch_graph
    return

  graph = graph or func_graph.FuncGraph('keras_scratch_graph')
  try:
    _CURRENT_SCRATCH_GRAPH.graph = graph
    yield graph
  finally:
    _CURRENT_SCRATCH_GRAPH.graph = None

@keras_export(v1=['keras.backend.set_session'])
def set_session(session):
  global _SESSION
  _SESSION.session = session

def get_default_session_config():
  if os.environ.get('OMP_NUM_THREADS'):
    logging.warning(
        'OMP_NUM_THREADS is no longer used by the default Keras config. '
        'To configure the number of threads, use tf.config.threading APIs.')

  config = context.context().config
  config.allow_soft_placement = True

  return config

def get_default_graph_uid_map():
  graph = ops.get_default_graph()
  name_uid_map = PER_GRAPH_OBJECT_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections.defaultdict(int)
    PER_GRAPH_OBJECT_NAME_UIDS[graph] = name_uid_map
  return name_uid_map

class _TfDeviceCaptureOp(object):

  def __init__(self):
    self.device = None

  def _set_device(self, device):
    if tfdev.is_device_spec(device):
      device = device.to_string()
    self.device = device

  def _set_device_from_string(self, device_str):
    self.device = device_str

def _get_current_tf_device():
  graph = get_graph()
  op = _TfDeviceCaptureOp()
  graph._apply_device_functions(op)
  return tfdev.DeviceSpec.from_string(op.device)

def _is_current_explicit_device(device_type):
  device_type = device_type.upper()
  if device_type not in ['CPU', 'GPU']:
    raise ValueError('`device_type` should be either "CPU" or "GPU".')
  device = _get_current_tf_device()
  return device is not None and device.device_type == device_type.upper()

def _get_available_gpus():
  if ops.executing_eagerly_outside_functions():
    return [d.name for d in config.list_logical_devices('GPU')]

  global _LOCAL_DEVICES
  if _LOCAL_DEVICES is None:
    _LOCAL_DEVICES = get_session().list_devices()
  return [x.name for x in _LOCAL_DEVICES if x.device_type == 'GPU']

def _has_nchw_support():
  explicitly_on_cpu = _is_current_explicit_device('CPU')
  gpus_available = bool(_get_available_gpus())
  return not explicitly_on_cpu and gpus_available

def _constant_to_tensor(x, dtype):
  return constant_op.constant(x, dtype=dtype)

def _to_tensor(x, dtype):
  return ops.convert_to_tensor_v2(x, dtype=dtype)

@keras_export('keras.backend.is_sparse')
def is_sparse(tensor):
  spec = getattr(tensor, '_type_spec', None)
  if spec is not None:
    return isinstance(spec, sparse_tensor.SparseTensorSpec)
  return isinstance(tensor, sparse_tensor.SparseTensor)

@keras_export('keras.backend.to_dense')
@dispatch.add_dispatch_support
def to_dense(tensor):
  if is_sparse(tensor):
    return sparse_ops.sparse_tensor_to_dense(tensor)
  else:
    return tensor

@keras_export('keras.backend.name_scope', v1=[])
def name_scope(name):
  return ops.name_scope_v2(name)

keras_export(v1=['keras.backend.name_scope'])(ops.name_scope_v1)

@keras_export('keras.backend.variable')
def variable(value, dtype=None, name=None, constraint=None):
  if dtype is None:
    dtype = floatx()
  if hasattr(value, 'tocoo'):
    sparse_coo = value.tocoo()
    indices = np.concatenate((np.expand_dims(sparse_coo.row, 1), np.expand_dims(
        sparse_coo.col, 1)), 1)
    v = sparse_tensor.SparseTensor(
        indices=indices, values=sparse_coo.data, dense_shape=sparse_coo.shape)
    v._keras_shape = sparse_coo.shape
    return v
  v = variables_module.Variable(
      value,
      dtype=dtypes_module.as_dtype(dtype),
      name=name,
      constraint=constraint)
  if isinstance(value, np.ndarray):
    v._keras_shape = value.shape
  elif hasattr(value, 'shape'):
    v._keras_shape = int_shape(value)
  track_variable(v)
  return v

def track_tf_optimizer(tf_optimizer):
  if context.executing_eagerly():
    return
  optimizers = _GRAPH_TF_OPTIMIZERS[None]
  optimizers.add(tf_optimizer)

def track_variable(v):
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else get_graph()
  _GRAPH_VARIABLES[graph].add(v)

def unique_object_name(name,
                       name_uid_map=None,
                       avoid_names=None,
                       namespace='',
                       zero_based=False):
  if name_uid_map is None:
    name_uid_map = get_default_graph_uid_map()
  if avoid_names is None:
    avoid_names = set()
  proposed_name = None
  while proposed_name is None or proposed_name in avoid_names:
    name_key = (namespace, name)
    if zero_based:
      number = name_uid_map[name_key]
      if number:
        proposed_name = name + '_' + str(number)
      else:
        proposed_name = name
      name_uid_map[name_key] += 1
    else:
      name_uid_map[name_key] += 1
      proposed_name = name + '_' + str(name_uid_map[name_key])
  return proposed_name

def _get_variables(graph=None):
  assert not context.executing_eagerly()
  variables = _GRAPH_VARIABLES[graph]
  for opt in _GRAPH_TF_OPTIMIZERS[graph]:
    variables.update(opt.optimizer.variables())
  return variables

def _initialize_variables(session):
  variables = _get_variables(get_graph())
  candidate_vars = []
  for v in variables:
    if not getattr(v, '_keras_initialized', False):
      candidate_vars.append(v)
  if candidate_vars:
    is_initialized = session.run(
        [variables_module.is_variable_initialized(v) for v in candidate_vars])
    should_be_initialized = [
        (not is_initialized[n]) and v.initializer is not None
        for n, v in enumerate(candidate_vars)]
    uninitialized_vars = []
    for flag, v in zip(should_be_initialized, candidate_vars):
      if flag:
        uninitialized_vars.append(v)
      v._keras_initialized = True
    if uninitialized_vars:
      session.run(variables_module.variables_initializer(uninitialized_vars))

@keras_export('keras.backend.constant')
@dispatch.add_dispatch_support
def constant(value, dtype=None, shape=None, name=None):
  if dtype is None:
    dtype = floatx()

  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)

@keras_export('keras.backend.is_keras_tensor')
def is_keras_tensor(x):
  if not isinstance(x,
                    (ops.Tensor, variables_module.Variable,
                     sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor,
                     keras_tensor.KerasTensor)):
    raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                     '`. Expected a symbolic tensor instance.')
  if keras_tensor.keras_tensors_enabled():
    return isinstance(x, keras_tensor.KerasTensor)
  return hasattr(x, '_keras_history')

@keras_export('keras.backend.placeholder')
def placeholder(shape=None,
                ndim=None,
                dtype=None,
                sparse=False,
                name=None,
                ragged=False):
  if sparse and ragged:
    raise ValueError(
        'Cannot set both sparse and ragged to True when creating a placeholder.'
    )
  if dtype is None:
    dtype = floatx()
  if not shape:
    if ndim:
      shape = (None,) * ndim
  if keras_tensor.keras_tensors_enabled():
    spec = tensor_spec.TensorSpec(
        shape=shape, dtype=dtype, name=name)
    if sparse:
      spec = sparse_tensor.SparseTensorSpec(
          shape=shape, dtype=dtype)
    elif ragged:
      ragged_rank = 0
      for i in range(1, len(shape)):
        if shape[i] is None or (
            hasattr(shape[i], 'value') and
            shape[i].value is None):
          ragged_rank = i
      spec = ragged_tensor.RaggedTensorSpec(
          shape=shape, dtype=dtype, ragged_rank=ragged_rank)

    x = keras_tensor.KerasTensor(spec, name=name)
  else:
    with get_graph().as_default():
      if sparse:
        x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
      elif ragged:
        ragged_rank = 0
        for i in range(1, len(shape)):
          if shape[i] is None:
            ragged_rank = i
        type_spec = ragged_tensor.RaggedTensorSpec(
            shape=shape, dtype=dtype, ragged_rank=ragged_rank)
        def tensor_spec_to_placeholder(tensorspec):
          return array_ops.placeholder(tensorspec.dtype, tensorspec.shape)
        x = nest.map_structure(tensor_spec_to_placeholder, type_spec,
                               expand_composites=True)
      else:
        x = array_ops.placeholder(dtype, shape=shape, name=name)

  if context.executing_eagerly():
    from tensorflow.python.keras.engine import input_layer  
    x = input_layer.Input(tensor=x)
    if keras_tensor.keras_tensors_enabled():
      x._is_backend_placeholder = True

  return x

def is_placeholder(x):
  try:
    if keras_tensor.keras_tensors_enabled():
      return hasattr(x, '_is_backend_placeholder')
    if isinstance(x, composite_tensor.CompositeTensor):
      flat_components = nest.flatten(x, expand_composites=True)
      return py_any(is_placeholder(c) for c in flat_components)
    else:
      return x.op.type == 'Placeholder'
  except AttributeError:
    return False

@keras_export('keras.backend.shape')
@dispatch.add_dispatch_support
def shape(x):
  return array_ops.shape(x)

@keras_export('keras.backend.int_shape')
def int_shape(x):
  try:
    shape = x.shape
    if not isinstance(shape, tuple):
      shape = tuple(shape.as_list())
    return shape
  except ValueError:
    return None

@keras_export('keras.backend.ndim')
def ndim(x):
  dims = x.shape._dims
  if dims is not None:
    return len(dims)
  return None

@keras_export('keras.backend.dtype')
@dispatch.add_dispatch_support
def dtype(x):
  return x.dtype.base_dtype.name

@keras_export('keras.backend.eval')
def eval(x):
  return get_value(to_dense(x))

@keras_export('keras.backend.zeros')
def zeros(shape, dtype=None, name=None):
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.zeros(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    return v

@keras_export('keras.backend.ones')
@dispatch.add_dispatch_support
def ones(shape, dtype=None, name=None):
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.ones(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    return v

@keras_export('keras.backend.eye')
@dispatch.add_dispatch_support
def eye(size, dtype=None, name=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  return variable(linalg_ops.eye(size, dtype=tf_dtype), dtype, name)

@keras_export('keras.backend.zeros_like')
def zeros_like(x, dtype=None, name=None):
  return array_ops.zeros_like(x, dtype=dtype, name=name)

@keras_export('keras.backend.ones_like')
@dispatch.add_dispatch_support
def ones_like(x, dtype=None, name=None):
  return array_ops.ones_like(x, dtype=dtype, name=name)

def identity(x, name=None):
  return array_ops.identity(x, name=name)

@keras_export('keras.backend.random_uniform_variable')
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    seed = np.random.randint(10e8)
  value = init_ops.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)

@keras_export('keras.backend.random_normal_variable')
def random_normal_variable(shape, mean, scale, dtype=None, name=None,
                           seed=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    seed = np.random.randint(10e8)
  value = init_ops.random_normal_initializer(
      mean, scale, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)

@keras_export('keras.backend.count_params')
def count_params(x):
  return np.prod(x.shape.as_list())

@keras_export('keras.backend.cast')
@dispatch.add_dispatch_support
def cast(x, dtype):
  return math_ops.cast(x, dtype)

@keras_export('keras.backend.update')
def update(x, new_x):
  return state_ops.assign(x, new_x)

@keras_export('keras.backend.update_add')
def update_add(x, increment):
  return state_ops.assign_add(x, increment)

@keras_export('keras.backend.update_sub')
def update_sub(x, decrement):
  return state_ops.assign_sub(x, decrement)

@keras_export('keras.backend.moving_average_update')
def moving_average_update(x, value, momentum):
  zero_debias = not tf2.enabled()
  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=zero_debias)

@keras_export('keras.backend.dot')
@dispatch.add_dispatch_support
def dot(x, y):
  if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
    x_shape = []
    for i, s in zip(int_shape(x), array_ops.unstack(array_ops.shape(x))):
      if i is not None:
        x_shape.append(i)
      else:
        x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), array_ops.unstack(array_ops.shape(y))):
      if i is not None:
        y_shape.append(i)
      else:
        y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = array_ops.reshape(x, [-1, x_shape[-1]])
    yt = array_ops.reshape(
        array_ops.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return array_ops.reshape(
        math_ops.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
  if is_sparse(x):
    out = sparse_ops.sparse_tensor_dense_matmul(x, y)
  else:
    out = math_ops.matmul(x, y)
  return out

@keras_export('keras.backend.batch_dot')
@dispatch.add_dispatch_support
def batch_dot(x, y, axes=None):
  x_shape = int_shape(x)
  y_shape = int_shape(y)

  x_ndim = len(x_shape)
  y_ndim = len(y_shape)

  if x_ndim < 2 or y_ndim < 2:
    raise ValueError('Cannot do batch_dot on inputs '
                     'with rank < 2. '
                     'Received inputs with shapes ' +
                     str(x_shape) + ' and ' +
                     str(y_shape) + '.')

  x_batch_size = x_shape[0]
  y_batch_size = y_shape[0]

  if x_batch_size is not None and y_batch_size is not None:
    if x_batch_size != y_batch_size:
      raise ValueError('Cannot do batch_dot on inputs '
                       'with different batch sizes. '
                       'Received inputs with shapes ' +
                       str(x_shape) + ' and ' +
                       str(y_shape) + '.')
  if isinstance(axes, int):
    axes = [axes, axes]

  if axes is None:
    if y_ndim == 2:
      axes = [x_ndim - 1, y_ndim - 1]
    else:
      axes = [x_ndim - 1, y_ndim - 2]

  if py_any(isinstance(a, (list, tuple)) for a in axes):
    raise ValueError('Multiple target dimensions are not supported. ' +
                     'Expected: None, int, (int, int), ' +
                     'Provided: ' + str(axes))

  axes = list(axes)

  if axes[0] < 0:
    axes[0] += x_ndim
  if axes[1] < 0:
    axes[1] += y_ndim

  if 0 in axes:
    raise ValueError('Cannot perform batch_dot over axis 0. '
                     'If your inputs are not batched, '
                     'add a dummy batch dimension to your '
                     'inputs using K.expand_dims(x, 0)')
  a0, a1 = axes
  d1 = x_shape[a0]
  d2 = y_shape[a1]

  if d1 is not None and d2 is not None and d1 != d2:
    raise ValueError('Cannot do batch_dot on inputs with shapes ' +
                     str(x_shape) + ' and ' + str(y_shape) +
                     ' with axes=' + str(axes) + '. x.shape[%d] != '
                     'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))

  orig_x_ndim = x_ndim
  orig_y_ndim = y_ndim

  if x_ndim == 2:
    x = array_ops.expand_dims(x, 1)
    a0 += 1
    x_ndim += 1
  if y_ndim == 2:
    y = array_ops.expand_dims(y, 2)
    y_ndim += 1

  if a0 != x_ndim - 1:
    pattern = list(range(x_ndim))
    for i in range(a0, x_ndim - 1):
      pattern[i] = pattern[i + 1]
    pattern[-1] = a0
    x = array_ops.transpose(x, pattern)

  if a1 != 1:
    pattern = list(range(y_ndim))
    for i in range(a1, 1, -1):
      pattern[i] = pattern[i - 1]
    pattern[1] = a1
    y = array_ops.transpose(y, pattern)

  if x_ndim > 3:
    x_shape = shape(x)
    x_mid_dims = x_shape[1:-1]
    x_squashed_shape = array_ops.stack(
        [x_shape[0], -1, x_shape[-1]])
    x = array_ops.reshape(x, x_squashed_shape)
    x_squashed = True
  else:
    x_squashed = False

  if y_ndim > 3:
    y_shape = shape(y)
    y_trail_dims = y_shape[2:]
    y_squashed_shape = array_ops.stack(
        [y_shape[0], y_shape[1], -1])
    y = array_ops.reshape(y, y_squashed_shape)
    y_squashed = True
  else:
    y_squashed = False

  result = math_ops.matmul(x, y)

  output_shape = array_ops.shape(result)
  do_reshape = False

  if x_squashed:
    output_shape = array_ops.concat(
        [output_shape[:1],
         x_mid_dims,
         output_shape[-1:]], 0)
    do_reshape = True

  if y_squashed:
    output_shape = array_ops.concat([output_shape[:-1], y_trail_dims], 0)
    do_reshape = True

  if do_reshape:
    result = array_ops.reshape(result, output_shape)

  if orig_x_ndim == 2:
    result = array_ops.squeeze(result, 1)
  elif orig_y_ndim == 2:
    result = array_ops.squeeze(result, -1)

  return result

@keras_export('keras.backend.transpose')
@dispatch.add_dispatch_support
def transpose(x):
  return array_ops.transpose(x)

@keras_export('keras.backend.gather')
@dispatch.add_dispatch_support
def gather(reference, indices):
  return array_ops.gather(reference, indices)

@keras_export('keras.backend.max')
@dispatch.add_dispatch_support
def max(x, axis=None, keepdims=False):
  return math_ops.reduce_max(x, axis, keepdims)

@keras_export('keras.backend.min')
@dispatch.add_dispatch_support
def min(x, axis=None, keepdims=False):
  return math_ops.reduce_min(x, axis, keepdims)

@keras_export('keras.backend.sum')
@dispatch.add_dispatch_support
def sum(x, axis=None, keepdims=False):
  return math_ops.reduce_sum(x, axis, keepdims)

@keras_export('keras.backend.prod')
@dispatch.add_dispatch_support
def prod(x, axis=None, keepdims=False):
  return math_ops.reduce_prod(x, axis, keepdims)

@keras_export('keras.backend.cumsum')
@dispatch.add_dispatch_support
def cumsum(x, axis=0):
  return math_ops.cumsum(x, axis=axis)

@keras_export('keras.backend.cumprod')
@dispatch.add_dispatch_support
def cumprod(x, axis=0):
  return math_ops.cumprod(x, axis=axis)

@keras_export('keras.backend.var')
def var(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_variance(x, axis=axis, keepdims=keepdims)

@keras_export('keras.backend.std')
@dispatch.add_dispatch_support
def std(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_std(x, axis=axis, keepdims=keepdims)

@keras_export('keras.backend.mean')
@dispatch.add_dispatch_support
def mean(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, axis, keepdims)

@keras_export('keras.backend.any')
@dispatch.add_dispatch_support
def any(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, axis, keepdims)

@keras_export('keras.backend.all')
@dispatch.add_dispatch_support
def all(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, axis, keepdims)

@keras_export('keras.backend.argmax')
@dispatch.add_dispatch_support
def argmax(x, axis=-1):
  return math_ops.argmax(x, axis)

@keras_export('keras.backend.argmin')
@dispatch.add_dispatch_support
def argmin(x, axis=-1):
  return math_ops.argmin(x, axis)

@keras_export('keras.backend.square')
@dispatch.add_dispatch_support
def square(x):
  return math_ops.square(x)

@keras_export('keras.backend.abs')
@dispatch.add_dispatch_support
def abs(x):
  return math_ops.abs(x)

@keras_export('keras.backend.sqrt')
@dispatch.add_dispatch_support
def sqrt(x):
  zero = _constant_to_tensor(0., x.dtype.base_dtype)
  inf = _constant_to_tensor(np.inf, x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, inf)
  return math_ops.sqrt(x)

@keras_export('keras.backend.exp')
@dispatch.add_dispatch_support
def exp(x):
  return math_ops.exp(x)

@keras_export('keras.backend.log')
@dispatch.add_dispatch_support
def log(x):
  return math_ops.log(x)

def logsumexp(x, axis=None, keepdims=False):
  return math_ops.reduce_logsumexp(x, axis, keepdims)

@keras_export('keras.backend.round')
@dispatch.add_dispatch_support
def round(x):
  return math_ops.round(x)

@keras_export('keras.backend.sign')
@dispatch.add_dispatch_support
def sign(x):
  return math_ops.sign(x)

@keras_export('keras.backend.pow')
@dispatch.add_dispatch_support
def pow(x, a):
  return math_ops.pow(x, a)

@keras_export('keras.backend.clip')
@dispatch.add_dispatch_support
def clip(x, min_value, max_value):
  if (isinstance(min_value, (int, float)) and
      isinstance(max_value, (int, float))):
    if max_value < min_value:
      max_value = min_value
  if min_value is None:
    min_value = -np.inf
  if max_value is None:
    max_value = np.inf
  return clip_ops.clip_by_value(x, min_value, max_value)

@keras_export('keras.backend.equal')
@dispatch.add_dispatch_support
def equal(x, y):
  return math_ops.equal(x, y)

@keras_export('keras.backend.not_equal')
@dispatch.add_dispatch_support
def not_equal(x, y):
  return math_ops.not_equal(x, y)

@keras_export('keras.backend.greater')
@dispatch.add_dispatch_support
def greater(x, y):
  return math_ops.greater(x, y)

@keras_export('keras.backend.greater_equal')
@dispatch.add_dispatch_support
def greater_equal(x, y):
  return math_ops.greater_equal(x, y)

@keras_export('keras.backend.less')
@dispatch.add_dispatch_support
def less(x, y):
  return math_ops.less(x, y)

@keras_export('keras.backend.less_equal')
@dispatch.add_dispatch_support
def less_equal(x, y):
  return math_ops.less_equal(x, y)

@keras_export('keras.backend.maximum')
@dispatch.add_dispatch_support
def maximum(x, y):
  return math_ops.maximum(x, y)

@keras_export('keras.backend.minimum')
@dispatch.add_dispatch_support
def minimum(x, y):
  return math_ops.minimum(x, y)

@keras_export('keras.backend.sin')
@dispatch.add_dispatch_support
def sin(x):
  return math_ops.sin(x)

@keras_export('keras.backend.cos')
@dispatch.add_dispatch_support
def cos(x):
  return math_ops.cos(x)

def _regular_normalize_batch_in_training(x,
                                         gamma,
                                         beta,
                                         reduction_axes,
                                         epsilon=1e-3):
  mean, var = nn.moments(x, reduction_axes, None, None, False)
  normed = nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
  return normed, mean, var

def _broadcast_normalize_batch_in_training(x,
                                           gamma,
                                           beta,
                                           reduction_axes,
                                           epsilon=1e-3):
  mean, var = nn.moments(x, reduction_axes, None, None, False)
  target_shape = []
  for axis in range(ndim(x)):
    if axis in reduction_axes:
      target_shape.append(1)
    else:
      target_shape.append(array_ops.shape(x)[axis])
  target_shape = array_ops.stack(target_shape)

  broadcast_mean = array_ops.reshape(mean, target_shape)
  broadcast_var = array_ops.reshape(var, target_shape)
  if gamma is None:
    broadcast_gamma = None
  else:
    broadcast_gamma = array_ops.reshape(gamma, target_shape)
  if beta is None:
    broadcast_beta = None
  else:
    broadcast_beta = array_ops.reshape(beta, target_shape)

  normed = nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                  broadcast_beta, broadcast_gamma, epsilon)
  return normed, mean, var

def _fused_normalize_batch_in_training(x,
                                       gamma,
                                       beta,
                                       reduction_axes,
                                       epsilon=1e-3):
  if list(reduction_axes) == [0, 1, 2]:
    normalization_axis = 3
    tf_data_format = 'NHWC'
  else:
    normalization_axis = 1
    tf_data_format = 'NCHW'

  if gamma is None:
    gamma = constant_op.constant(
        1.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])
  if beta is None:
    beta = constant_op.constant(
        0.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])

  return nn.fused_batch_norm(
      x, gamma, beta, epsilon=epsilon, data_format=tf_data_format)

@keras_export('keras.backend.normalize_batch_in_training')
def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
  if ndim(x) == 4 and list(reduction_axes) in [[0, 1, 2], [0, 2, 3]]:
    if not _has_nchw_support() and list(reduction_axes) == [0, 2, 3]:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    return _fused_normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=epsilon)
  else:
    if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
      return _regular_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    else:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)

@keras_export('keras.backend.batch_normalization')
@dispatch.add_dispatch_support
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
  if ndim(x) == 4:
    if axis == 1 or axis == -3:
      tf_data_format = 'NCHW'
    elif axis == 3 or axis == -1:
      tf_data_format = 'NHWC'
    else:
      tf_data_format = None

    if (tf_data_format == 'NHWC' or
        tf_data_format == 'NCHW' and _has_nchw_support()):
      if ndim(mean) > 1:
        mean = array_ops.reshape(mean, [-1])
      if ndim(var) > 1:
        var = array_ops.reshape(var, [-1])
      if beta is None:
        beta = zeros_like(mean)
      elif ndim(beta) > 1:
        beta = array_ops.reshape(beta, [-1])
      if gamma is None:
        gamma = ones_like(mean)
      elif ndim(gamma) > 1:
        gamma = array_ops.reshape(gamma, [-1])
    y, _, _ = nn.fused_batch_norm(
        x,
        gamma,
        beta,
        epsilon=epsilon,
        mean=mean,
        variance=var,
        data_format=tf_data_format,
        is_training=False
    )
    return y
  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

@keras_export('keras.backend.concatenate')
@dispatch.add_dispatch_support
def concatenate(tensors, axis=-1):
  if axis < 0:
    rank = ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if py_all(is_sparse(x) for x in tensors):
    return sparse_ops.sparse_concat(axis, tensors)
  elif py_all(isinstance(x, ragged_tensor.RaggedTensor) for x in tensors):
    return ragged_concat_ops.concat(tensors, axis)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)

@keras_export('keras.backend.reshape')
@dispatch.add_dispatch_support
def reshape(x, shape):
  return array_ops.reshape(x, shape)

@keras_export('keras.backend.permute_dimensions')
@dispatch.add_dispatch_support
def permute_dimensions(x, pattern):
  return array_ops.transpose(x, perm=pattern)

@keras_export('keras.backend.resize_images')
@dispatch.add_dispatch_support
def resize_images(x, height_factor, width_factor, data_format,
                  interpolation='nearest'):
  if data_format == 'channels_first':
    rows, cols = 2, 3
  elif data_format == 'channels_last':
    rows, cols = 1, 2
  else:
    raise ValueError('Invalid `data_format` argument: %s' % (data_format,))

  original_shape = int_shape(x)
  new_shape = array_ops.shape(x)[rows:cols + 1]
  new_shape *= constant_op.constant(
      np.array([height_factor, width_factor], dtype='int32'))

  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 2, 3, 1])
  if interpolation == 'nearest':
    x = image_ops.resize_images_v2(
        x, new_shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
  elif interpolation == 'bilinear':
    x = image_ops.resize_images_v2(x, new_shape,
                                   method=image_ops.ResizeMethod.BILINEAR)
  else:
    raise ValueError('interpolation should be one '
                     'of "nearest" or "bilinear".')
  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 3, 1, 2])

  if original_shape[rows] is None:
    new_height = None
  else:
    new_height = original_shape[rows] * height_factor

  if original_shape[cols] is None:
    new_width = None
  else:
    new_width = original_shape[cols] * width_factor

  if data_format == 'channels_first':
    output_shape = (None, None, new_height, new_width)
  else:
    output_shape = (None, new_height, new_width, None)
  x.set_shape(output_shape)
  return x

@keras_export('keras.backend.resize_volumes')
@dispatch.add_dispatch_support
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
  if data_format == 'channels_first':
    output = repeat_elements(x, depth_factor, axis=2)
    output = repeat_elements(output, height_factor, axis=3)
    output = repeat_elements(output, width_factor, axis=4)
    return output
  elif data_format == 'channels_last':
    output = repeat_elements(x, depth_factor, axis=1)
    output = repeat_elements(output, height_factor, axis=2)
    output = repeat_elements(output, width_factor, axis=3)
    return output
  else:
    raise ValueError('Invalid data_format: ' + str(data_format))

@keras_export('keras.backend.repeat_elements')
@dispatch.add_dispatch_support
def repeat_elements(x, rep, axis):
  x_shape = x.shape.as_list()
  if x_shape[axis] is not None:
    splits = array_ops.split(value=x,
                             num_or_size_splits=x_shape[axis],
                             axis=axis)
    x_rep = [s for s in splits for _ in range(rep)]
    return concatenate(x_rep, axis)

  auxiliary_axis = axis + 1
  x_shape = array_ops.shape(x)
  x_rep = array_ops.expand_dims(x, axis=auxiliary_axis)
  reps = np.ones(len(x.shape) + 1)
  reps[auxiliary_axis] = rep
  x_rep = array_ops.tile(x_rep, reps)

  reps = np.delete(reps, auxiliary_axis)
  reps[axis] = rep
  reps = array_ops.constant(reps, dtype='int32')
  x_shape *= reps
  x_rep = array_ops.reshape(x_rep, x_shape)

  x_shape = x.shape.as_list()
  x_rep.set_shape(x_shape)
  x_rep._keras_shape = tuple(x_shape)
  return x_rep

@keras_export('keras.backend.repeat')
@dispatch.add_dispatch_support
def repeat(x, n):
  assert ndim(x) == 2
  x = array_ops.expand_dims(x, 1)
  pattern = array_ops.stack([1, n, 1])
  return array_ops.tile(x, pattern)

@keras_export('keras.backend.arange')
@dispatch.add_dispatch_support
def arange(start, stop=None, step=1, dtype='int32'):
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result

@keras_export('keras.backend.tile')
@dispatch.add_dispatch_support
def tile(x, n):
  if isinstance(n, int):
    n = [n]
  return array_ops.tile(x, n)

@keras_export('keras.backend.flatten')
@dispatch.add_dispatch_support
def flatten(x):
  return array_ops.reshape(x, [-1])

@keras_export('keras.backend.batch_flatten')
@dispatch.add_dispatch_support
def batch_flatten(x):
  x = array_ops.reshape(x, array_ops.stack([-1, prod(shape(x)[1:])]))
  return x

@keras_export('keras.backend.expand_dims')
@dispatch.add_dispatch_support
def expand_dims(x, axis=-1):
  return array_ops.expand_dims(x, axis)

@keras_export('keras.backend.squeeze')
@dispatch.add_dispatch_support
def squeeze(x, axis):
  return array_ops.squeeze(x, [axis])

@keras_export('keras.backend.temporal_padding')
@dispatch.add_dispatch_support
def temporal_padding(x, padding=(1, 1)):
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return array_ops.pad(x, pattern)

@keras_export('keras.backend.spatial_2d_padding')
@dispatch.add_dispatch_support
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
  assert len(padding) == 2
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
  else:
    pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
  return array_ops.pad(x, pattern)

@keras_export('keras.backend.spatial_3d_padding')
@dispatch.add_dispatch_support
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
  assert len(padding) == 3
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  assert len(padding[2]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]]
  else:
    pattern = [[0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0],
                                                padding[2][1]], [0, 0]]
  return array_ops.pad(x, pattern)

@keras_export('keras.backend.stack')
@dispatch.add_dispatch_support
def stack(x, axis=0):
  return array_ops.stack(x, axis=axis)

@keras_export('keras.backend.one_hot')
@dispatch.add_dispatch_support
def one_hot(indices, num_classes):
  return array_ops.one_hot(indices, depth=num_classes, axis=-1)

@keras_export('keras.backend.reverse')
@dispatch.add_dispatch_support
def reverse(x, axes):
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)

_VALUE_SET_CODE_STRING = [3:]  

@keras_export('keras.backend.get_value')
def get_value(x):
  if not tensor_util.is_tensor(x):
    return x
  if context.executing_eagerly() or isinstance(x, ops.EagerTensor):
    return x.numpy()
  if not getattr(x, '_in_graph_mode', True):
    with context.eager_mode():
      return x.numpy()

  if ops.executing_eagerly_outside_functions():
    return eval_in_eager_or_function(x)

  with x.graph.as_default():
    return x.eval(session=get_session((x,)))

@keras_export('keras.backend.batch_get_value')
@dispatch.add_dispatch_support
def batch_get_value(tensors):
  if context.executing_eagerly():
    return [x.numpy() for x in tensors]
  elif ops.inside_function():  
    raise RuntimeError('Cannot get value inside Tensorflow graph function.')
  if tensors:
    return get_session(tensors).run(tensors)
  else:
    return []

@keras_export('keras.backend.set_value')
def set_value(x, value):
  value = np.asarray(value, dtype=dtype(x))
  if ops.executing_eagerly_outside_functions():
    x.assign(value)
  else:
    with get_graph().as_default():
      tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
      if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
      else:
        placeholder_shape = tensor_shape.TensorShape([None] * value.ndim)
        assign_placeholder = array_ops.placeholder(
            tf_dtype, shape=placeholder_shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      get_session().run(assign_op, feed_dict={assign_placeholder: value})

@keras_export('keras.backend.batch_set_value')
@dispatch.add_dispatch_support
def batch_set_value(tuples):
  if ops.executing_eagerly_outside_functions():
    for x, value in tuples:
      x.assign(np.asarray(value, dtype=dtype(x)))
  else:
    with get_graph().as_default():
      if tuples:
        assign_ops = []
        feed_dict = {}
        for x, value in tuples:
          value = np.asarray(value, dtype=dtype(x))
          tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
          if hasattr(x, '_assign_placeholder'):
            assign_placeholder = x._assign_placeholder
            assign_op = x._assign_op
          else:
            placeholder_shape = tensor_shape.TensorShape([None] * value.ndim)
            assign_placeholder = array_ops.placeholder(
                tf_dtype, shape=placeholder_shape)
            assign_op = x.assign(assign_placeholder)
            x._assign_placeholder = assign_placeholder
            x._assign_op = assign_op
          assign_ops.append(assign_op)
          feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)

get_value.__doc__ = get_value.__doc__.format(snippet=_VALUE_SET_CODE_STRING)
set_value.__doc__ = set_value.__doc__.format(snippet=_VALUE_SET_CODE_STRING)

@keras_export('keras.backend.print_tensor')
@dispatch.add_dispatch_support
def print_tensor(x, message=''):
  if isinstance(x, ops.Tensor) and hasattr(x, 'graph'):
    with get_graph().as_default():
      op = logging_ops.print_v2(message, x, output_stream=sys.stdout)
      with ops.control_dependencies([op]):
        return array_ops.identity(x)
  else:
    logging_ops.print_v2(message, x, output_stream=sys.stdout)
    return x

class GraphExecutionFunction(object):

  def __init__(self, inputs, outputs, updates=None, name=None,
               **session_kwargs):
    updates = updates or []
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a Keras backend function '
                      'should be a list or tuple.')

    self._inputs_structure = inputs
    self.inputs = nest.flatten(inputs, expand_composites=True)
    self._outputs_structure = outputs
    self.outputs = cast_variables_to_tensor(
        nest.flatten(outputs, expand_composites=True))
    with ops.control_dependencies([self.outputs[0]]):
      updates_ops = []
      for update in updates:
        if isinstance(update, tuple):
          p, new_p = update
          updates_ops.append(state_ops.assign(p, new_p))
        else:
          updates_ops.append(update)
      self.updates_op = control_flow_ops.group(*updates_ops)
    self.name = name
    self.feed_dict = session_kwargs.pop('feed_dict', None)
    self.fetches = session_kwargs.pop('fetches', [])
    if not isinstance(self.fetches, list):
      self.fetches = [self.fetches]
    self.run_options = session_kwargs.pop('options', None)
    self.run_metadata = session_kwargs.pop('run_metadata', None)
    self.fetches = [array_ops.identity(x) for x in self.fetches]
    self.session_kwargs = session_kwargs
    self.fetch_callbacks = {}

    if session_kwargs:
      raise ValueError('Some keys in session_kwargs are not supported at this '
                       'time: %s' % (session_kwargs.keys(),))

    self._callable_fn = None
    self._feed_arrays = None
    self._feed_symbols = None
    self._symbol_vals = None
    self._fetches = None
    self._session = None

  def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
    callable_opts = config_pb2.CallableOptions()
    for x in feed_arrays:
      callable_opts.feed.append(x.name)
    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        callable_opts.feed.append(key.name)
    for x, y in zip(feed_symbols, symbol_vals):
      connection = callable_opts.tensor_connection.add()
      if x.dtype != y.dtype:
        y = math_ops.cast(y, dtype=x.dtype)
      from_tensor = ops._as_graph_element(y)
      if from_tensor is None:
        from_tensor = y
      connection.from_tensor = from_tensor.name  
      connection.to_tensor = x.name  
    for x in self.outputs + self.fetches:
      callable_opts.fetch.append(x.name)
    callable_opts.target.append(self.updates_op.name)
    if self.run_options:
      callable_opts.run_options.CopyFrom(self.run_options)
    callable_fn = session._make_callable_from_options(callable_opts)
    self._callable_fn = callable_fn
    self._feed_arrays = feed_arrays
    self._feed_symbols = feed_symbols
    self._symbol_vals = symbol_vals
    self._fetches = list(self.fetches)
    self._session = session

  def _call_fetch_callbacks(self, fetches_output):
    for fetch, output in zip(self._fetches, fetches_output):
      if fetch in self.fetch_callbacks:
        self.fetch_callbacks[fetch](output)

  def _eval_if_composite(self, tensor):
    if isinstance(tensor, composite_tensor.CompositeTensor):
      return self._session.run(tensor)
    else:
      return tensor

  def __call__(self, inputs):
    inputs = nest.flatten(inputs, expand_composites=True)

    session = get_session(inputs)
    feed_arrays = []
    array_vals = []
    feed_symbols = []
    symbol_vals = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        continue

      if tensor_util.is_tensor(value):
        feed_symbols.append(tensor)
        symbol_vals.append(value)
      else:
        feed_arrays.append(tensor)
        tensor_type = dtypes_module.as_dtype(tensor.dtype)
        array_vals.append(np.asarray(value,
                                     dtype=tensor_type.as_numpy_dtype))

    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        array_vals.append(
            np.asarray(self.feed_dict[key], dtype=key.dtype.base_dtype.name))

    if (self._callable_fn is None or feed_arrays != self._feed_arrays or
        symbol_vals != self._symbol_vals or
        feed_symbols != self._feed_symbols or self.fetches != self._fetches or
        session != self._session):
      self._make_callable(feed_arrays, feed_symbols, symbol_vals, session)

    fetched = self._callable_fn(*array_vals,
                                run_metadata=self.run_metadata)
    self._call_fetch_callbacks(fetched[-len(self._fetches):])
    output_structure = nest.pack_sequence_as(
        self._outputs_structure,
        fetched[:len(self.outputs)],
        expand_composites=True)
    return nest.map_structure(self._eval_if_composite, output_structure)

def eval_in_eager_or_function(outputs):
  outputs_structure = outputs
  outputs = nest.flatten(outputs, expand_composites=True)

  graphs = {
      i.graph
      for i in nest.flatten([outputs])
      if hasattr(i, 'graph')
  }
  if len(graphs) > 1:
    raise ValueError('Cannot create an execution function which is comprised '
                     'of elements from multiple graphs.')

  source_graph = graphs.pop()

  with _scratch_graph() as exec_graph:
    global_graph = get_graph()
    if source_graph not in (exec_graph, global_graph):
      raise ValueError('Unknown graph. Aborting.')

    if source_graph is global_graph and exec_graph is not global_graph:
      init_tensors = outputs
      lifted_map = lift_to_graph.lift_to_graph(
          tensors=init_tensors,
          graph=exec_graph,
          sources=[],
          add_sources=True,
          handle_captures=True,
          base_graph=source_graph)

      outputs = [lifted_map[i] for i in outputs]

  with exec_graph.as_default():
    outputs = cast_variables_to_tensor(outputs)

    exec_graph.inputs = exec_graph.internal_captures
    exec_graph.outputs = outputs
    graph_fn = eager_function.ConcreteFunction(exec_graph)

  graph_fn._num_positional_args = 0
  graph_fn._arg_keywords = []

  outputs = graph_fn()

  return nest.pack_sequence_as(
      outputs_structure,
      [x._numpy() for x in outputs],  
      expand_composites=True)

@keras_export('keras.backend.function')
def function(inputs, outputs, updates=None, name=None, **kwargs):
  if ops.executing_eagerly_outside_functions():
    if kwargs:
      raise ValueError('Session keyword arguments are not supported during '
                       'eager execution. You passed: %s' % (kwargs,))
    if updates:
      raise ValueError('`updates` argument is not supported during '
                       'eager execution. You passed: %s' % (updates,))
    from tensorflow.python.keras import models  
    from tensorflow.python.keras.utils import tf_utils  
    model = models.Model(inputs=inputs, outputs=outputs)

    wrap_outputs = isinstance(outputs, list) and len(outputs) == 1
    def func(model_inputs):
      outs = model(model_inputs)
      if wrap_outputs:
        outs = [outs]
      return tf_utils.to_numpy_or_python_type(outs)
    return func

  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in ['inputs', 'outputs', 'updates', 'name']):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return GraphExecutionFunction(
      inputs, outputs, updates=updates, name=name, **kwargs)

@keras_export('keras.backend.gradients')
def gradients(loss, variables):
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)

@keras_export('keras.backend.stop_gradient')
@dispatch.add_dispatch_support
def stop_gradient(variables):
  if isinstance(variables, (list, tuple)):
    return map(array_ops.stop_gradient, variables)
  return array_ops.stop_gradient(variables)

@keras_export('keras.backend.rnn')
@dispatch.add_dispatch_support
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False):

  def swap_batch_timestep(input_t):
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if not time_major:
    inputs = nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  def _expand_mask(mask_t, input_t, fixed_dim=1):
    if nest.is_sequence(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if nest.is_sequence(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    def _process_single_input_t(input_t):
      input_t = array_ops.unstack(input_t)  
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_sequence(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp,
                                           tuple(states) + tuple(constants))
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where_v2(tiled_mask_t, output, prev_output)

        flat_states = nest.flatten(states)
        flat_new_states = nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            array_ops.where_v2(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where_v2(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = array_ops.where_v2(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, tuple(states) + tuple(constants))
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:  
    states = tuple(initial_states)

    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    output_time_zero, _ = step_function(
        input_time_zero, tuple(initial_states) + tuple(constants))
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    if (not context.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())):
      max_iterations = math_ops.reduce_max(input_length)
    else:
      max_iterations = None

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': max_iterations,
        'parallel_iterations': 32,
        'swap_memory': True,
    }
    if mask is not None:
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def masking_fn(time):
        return mask_ta.read(time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        tiled_mask_t = tuple(
            _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
            for o in flat_out)
        return tuple(
            array_ops.where_v2(m, o, fm)
            for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask))
    elif isinstance(input_length, ops.Tensor):
      if go_backwards:
        max_len = math_ops.reduce_max(input_length, axis=0)
        rev_input_length = math_ops.subtract(max_len - 1, input_length)

        def masking_fn(time):
          return math_ops.less(rev_input_length, time)
      else:

        def masking_fn(time):
          return math_ops.greater(input_length, time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        return tuple(
            array_ops.where(mask_t, o, zo)
            for (o, zo) in zip(flat_out, flat_mask))
    else:
      masking_fn = None

    if masking_fn is not None:
      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time, output_ta_t, prev_output, *states):
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = masking_fn(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask
                            else nest.flatten(prev_output))
        flat_new_output = compute_masked_output(mask_t, flat_output,
                                                flat_mask_output)

        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, ops.Tensor):
            new_state.set_shape(state.shape)
        flat_final_state = compute_masked_output(mask_t, flat_new_state,
                                                 flat_state)
        new_states = nest.pack_sequence_as(new_states, flat_final_state)

        output_ta_t = tuple(
            ta.write(time, out)
            for ta, out in zip(output_ta_t, flat_new_output))
        return (time + 1, output_ta_t,
                tuple(flat_new_output)) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta, flat_zero_output) + states,
          **while_loop_kwargs)
      new_states = final_outputs[3:]
    else:
      def _step(time, output_ta_t, *states):
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, ops.Tensor):
            new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        return (time + 1, output_ta_t) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta) + states,
          **while_loop_kwargs)
      new_states = final_outputs[2:]

    output_ta = final_outputs[1]

    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  def set_shape(output_):
    if isinstance(output_, ops.Tensor):
      shape = output_.shape.as_list()
      shape[0] = time_steps
      shape[1] = batch
      output_.set_shape(shape)
    return output_

  outputs = nest.map_structure(set_shape, outputs)

  if not time_major:
    outputs = nest.map_structure(swap_batch_timestep, outputs)

  return last_output, outputs, new_states

@keras_export('keras.backend.switch')
@dispatch.add_dispatch_support
def switch(condition, then_expression, else_expression):
  if condition.dtype != dtypes_module.bool:
    condition = math_ops.cast(condition, 'bool')
  cond_ndim = ndim(condition)
  if not cond_ndim:
    if not callable(then_expression):

      def then_expression_fn():
        return then_expression
    else:
      then_expression_fn = then_expression
    if not callable(else_expression):

      def else_expression_fn():
        return else_expression
    else:
      else_expression_fn = else_expression
    x = control_flow_ops.cond(condition, then_expression_fn, else_expression_fn)
  else:
    if callable(then_expression):
      then_expression = then_expression()
    if callable(else_expression):
      else_expression = else_expression()
    expr_ndim = ndim(then_expression)
    if cond_ndim > expr_ndim:
      raise ValueError('Rank of `condition` should be less than or'
                       ' equal to rank of `then_expression` and '
                       '`else_expression`. ndim(condition)=' + str(cond_ndim) +
                       ', ndim(then_expression)'
                       '=' + str(expr_ndim))
    if cond_ndim > 1:
      ndim_diff = expr_ndim - cond_ndim
      cond_shape = array_ops.concat(
          [array_ops.shape(condition), [1] * ndim_diff], axis=0)
      condition = array_ops.reshape(condition, cond_shape)
      expr_shape = array_ops.shape(then_expression)
      shape_diff = expr_shape - cond_shape
      tile_shape = array_ops.where_v2(shape_diff > 0, expr_shape,
                                      array_ops.ones_like(expr_shape))
      condition = array_ops.tile(condition, tile_shape)
    x = array_ops.where_v2(condition, then_expression, else_expression)
  return x

@keras_export('keras.backend.in_train_phase')
def in_train_phase(x, alt, training=None):
  from tensorflow.python.keras.engine import base_layer_utils  
  if training is None:
    training = base_layer_utils.call_context().training

  if training is None:
    training = learning_phase()

  if not tensor_util.is_tensor(training):
    if training == 1 or training is True:
      if callable(x):
        return x()
      else:
        return x

    elif training == 0 or training is False:
      if callable(alt):
        return alt()
      else:
        return alt

  x = switch(training, x, alt)
  return x

@keras_export('keras.backend.in_test_phase')
def in_test_phase(x, alt, training=None):
  return in_train_phase(alt, x, training=training)

@keras_export('keras.backend.relu')
@dispatch.add_dispatch_support
def relu(x, alpha=0., max_value=None, threshold=0):
  dtype = getattr(x, 'dtype', floatx())
  if alpha != 0.:
    if max_value is None and threshold == 0:
      return nn.leaky_relu(x, alpha=alpha)

    if threshold != 0:
      negative_part = nn.relu(-x + threshold)
    else:
      negative_part = nn.relu(-x)

  clip_max = max_value is not None

  if threshold != 0:
    x = x * math_ops.cast(math_ops.greater(x, threshold), dtype=dtype)
  elif max_value == 6:
    x = nn.relu6(x)
    clip_max = False
  else:
    x = nn.relu(x)

  if clip_max:
    max_value = _constant_to_tensor(max_value, x.dtype.base_dtype)
    zero = _constant_to_tensor(0, x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x

@keras_export('keras.backend.elu')
@dispatch.add_dispatch_support
def elu(x, alpha=1.):
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where_v2(x > 0, res, alpha * res)

@keras_export('keras.backend.softmax')
@dispatch.add_dispatch_support
def softmax(x, axis=-1):
  return nn.softmax(x, axis=axis)

@keras_export('keras.backend.softplus')
@dispatch.add_dispatch_support
def softplus(x):
  return nn.softplus(x)

@keras_export('keras.backend.softsign')
@dispatch.add_dispatch_support
def softsign(x):
  return nn.softsign(x)

@keras_export('keras.backend.categorical_crossentropy')
@dispatch.add_dispatch_support
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  target = ops.convert_to_tensor_v2(target)
  output = ops.convert_to_tensor_v2(output)

  target.shape.assert_is_compatible_with(output.shape)
  if from_logits:
    return nn.softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output, axis=axis)

  if (not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
      output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    return nn.softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output, axis=axis)

  output = output / math_ops.reduce_sum(output, axis, True)
  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
  return -math_ops.reduce_sum(target * math_ops.log(output), axis)

@keras_export('keras.backend.sparse_categorical_crossentropy')
@dispatch.add_dispatch_support
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  target = ops.convert_to_tensor_v2(target)
  output = ops.convert_to_tensor_v2(output)

  if (not from_logits and
      not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
      output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    from_logits = True

  if not from_logits:
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output)

  if isinstance(output.shape, (tuple, list)):
    output_rank = len(output.shape)
  else:
    output_rank = output.shape.ndims
  if output_rank is not None:
    axis %= output_rank
    if axis != output_rank - 1:
      permutation = list(
          itertools.chain(range(axis), range(axis + 1, output_rank), [axis]))
      output = array_ops.transpose(output, perm=permutation)
  elif axis != -1:
    raise ValueError(
        'Cannot compute sparse categorical crossentropy with `axis={}` on an '
        'output tensor with unknown rank'.format(axis))

  target = cast(target, 'int64')

  output_shape = array_ops.shape_v2(output)
  target_rank = target.shape.ndims

  update_shape = (
      target_rank is not None and output_rank is not None and
      target_rank != output_rank - 1)
  if update_shape:
    target = flatten(target)
    output = array_ops.reshape(output, [-1, output_shape[-1]])

  if py_any(_is_symbolic_tensor(v) for v in [target, output]):
    with get_graph().as_default():
      res = nn.sparse_softmax_cross_entropy_with_logits_v2(
          labels=target, logits=output)
  else:
    res = nn.sparse_softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output)

  if update_shape and output_rank >= 3:
    return array_ops.reshape(res, output_shape[:-1])
  else:
    return res

@keras_export('keras.backend.binary_crossentropy')
@dispatch.add_dispatch_support
def binary_crossentropy(target, output, from_logits=False):
  target = ops.convert_to_tensor_v2(target)
  output = ops.convert_to_tensor_v2(output)

  if from_logits:
    return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

  if (not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
      output.op.type == 'Sigmoid') and not hasattr(output, '_keras_history'):
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

  bce = target * math_ops.log(output + epsilon())
  bce += (1 - target) * math_ops.log(1 - output + epsilon())
  return -bce

@keras_export('keras.backend.sigmoid')
@dispatch.add_dispatch_support
def sigmoid(x):
  return nn.sigmoid(x)

@keras_export('keras.backend.hard_sigmoid')
@dispatch.add_dispatch_support
def hard_sigmoid(x):
  point_two = _constant_to_tensor(0.2, x.dtype.base_dtype)
  point_five = _constant_to_tensor(0.5, x.dtype.base_dtype)
  x = math_ops.mul(x, point_two)
  x = math_ops.add(x, point_five)
  x = clip_ops.clip_by_value(x, 0., 1.)
  return x

@keras_export('keras.backend.tanh')
@dispatch.add_dispatch_support
def tanh(x):
  return nn.tanh(x)

@keras_export('keras.backend.dropout')
@dispatch.add_dispatch_support
def dropout(x, level, noise_shape=None, seed=None):
  if seed is None:
    seed = np.random.randint(10e6)
  return nn.dropout_v2(x, rate=level, noise_shape=noise_shape, seed=seed)

@keras_export('keras.backend.l2_normalize')
@dispatch.add_dispatch_support
def l2_normalize(x, axis=None):
  return nn.l2_normalize(x, axis=axis)

@keras_export('keras.backend.in_top_k')
@dispatch.add_dispatch_support
def in_top_k(predictions, targets, k):
  return nn.in_top_k(predictions, targets, k)

def _preprocess_conv1d_input(x, data_format):
  tf_data_format = 'NWC'  
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 1))  
    else:
      tf_data_format = 'NCW'
  return x, tf_data_format

def _preprocess_conv2d_input(x, data_format, force_transpose=False):
  tf_data_format = 'NHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support() or force_transpose:
      x = array_ops.transpose(x, (0, 2, 3, 1))  
    else:
      tf_data_format = 'NCHW'
  return x, tf_data_format

def _preprocess_conv3d_input(x, data_format):
  tf_data_format = 'NDHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 3, 4, 1))
    else:
      tf_data_format = 'NCDHW'
  return x, tf_data_format

def _preprocess_padding(padding):
  if padding == 'same':
    padding = 'SAME'
  elif padding == 'valid':
    padding = 'VALID'
  else:
    raise ValueError('Invalid padding: ' + str(padding))
  return padding

@keras_export('keras.backend.conv1d')
@dispatch.add_dispatch_support
def conv1d(x,
           kernel,
           strides=1,
           padding='valid',
           data_format=None,
           dilation_rate=1):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = kernel.shape.as_list()
  if padding == 'causal':
    left_pad = dilation_rate * (kernel_shape[0] - 1)
    x = temporal_padding(x, (left_pad, 0))
    padding = 'valid'
  padding = _preprocess_padding(padding)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  
  return x

@keras_export('keras.backend.conv2d')
@dispatch.add_dispatch_support
def conv2d(x,
           kernel,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1)):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  
  return x

@keras_export('keras.backend.conv2d_transpose')
@dispatch.add_dispatch_support
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first' and dilation_rate != (1, 1):
    force_transpose = True
  else:
    force_transpose = False

  x, tf_data_format = _preprocess_conv2d_input(x, data_format, force_transpose)

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[1])
  if output_shape[0] is None:
    output_shape = (shape(x)[0],) + tuple(output_shape[1:])

  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  if dilation_rate == (1, 1):
    x = nn.conv2d_transpose(x, kernel, output_shape, strides,
                            padding=padding,
                            data_format=tf_data_format)
  else:
    assert dilation_rate[0] == dilation_rate[1]
    x = nn.atrous_conv2d_transpose(
        x,
        kernel,
        output_shape,
        rate=dilation_rate[0],
        padding=padding)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  
  return x

def separable_conv1d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=1,
                     padding='valid',
                     data_format=None,
                     dilation_rate=1):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if isinstance(strides, int):
    strides = (strides,)
  if isinstance(dilation_rate, int):
    dilation_rate = (dilation_rate,)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NWC':
    spatial_start_dim = 1
    strides = (1,) + strides * 2 + (1,)
  else:
    spatial_start_dim = 2
    strides = (1, 1) + strides * 2
  x = array_ops.expand_dims(x, spatial_start_dim)
  depthwise_kernel = array_ops.expand_dims(depthwise_kernel, 0)
  pointwise_kernel = array_ops.expand_dims(pointwise_kernel, 0)
  dilation_rate = (1,) + dilation_rate

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)

  x = array_ops.squeeze(x, [spatial_start_dim])

  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  

  return x

@keras_export('keras.backend.separable_conv2d')
@dispatch.add_dispatch_support
def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  
  return x

@keras_export('keras.backend.depthwise_conv2d')
@dispatch.add_dispatch_support
def depthwise_conv2d(x,
                     depthwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.depthwise_conv2d(
      x,
      depthwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  
  return x

@keras_export('keras.backend.conv3d')
@dispatch.add_dispatch_support
def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1, 1)):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x

def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(output_shape)

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[4], output_shape[1])
  if output_shape[0] is None:
    output_shape = (array_ops.shape(x)[0],) + tuple(output_shape[1:])
    output_shape = array_ops.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.conv3d_transpose(
      x,
      kernel,
      output_shape,
      strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x

@keras_export('keras.backend.pool2d')
@dispatch.add_dispatch_support
def pool2d(x,
           pool_size,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(pool_size) != 2:
    raise ValueError('`pool_size` must be a tuple of 2 integers.')
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  
  return x

@keras_export('keras.backend.pool3d')
@dispatch.add_dispatch_support
def pool3d(x,
           pool_size,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x

def local_conv(inputs,
               kernel,
               kernel_size,
               strides,
               output_shape,
               data_format=None):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = int_shape(kernel)
  feature_dim = kernel_shape[1]
  channels_out = kernel_shape[-1]
  ndims = len(output_shape)
  spatial_dimensions = list(range(ndims))

  xs = []
  output_axes_ticks = [range(axis_max) for axis_max in output_shape]
  for position in itertools.product(*output_axes_ticks):
    slices = [slice(None)]

    if data_format == 'channels_first':
      slices.append(slice(None))

    slices.extend(
        slice(position[d] * strides[d], position[d] * strides[d] +
              kernel_size[d]) for d in spatial_dimensions)

    if data_format == 'channels_last':
      slices.append(slice(None))

    xs.append(reshape(inputs[slices], (1, -1, feature_dim)))

  x_aggregate = concatenate(xs, axis=0)
  output = batch_dot(x_aggregate, kernel)
  output = reshape(output, output_shape + (-1, channels_out))

  if data_format == 'channels_first':
    permutation = [ndims, ndims + 1] + spatial_dimensions
  else:
    permutation = [ndims] + spatial_dimensions + [ndims + 1]

  return permute_dimensions(output, permutation)

@keras_export('keras.backend.local_conv1d')
@dispatch.add_dispatch_support
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  output_shape = (kernel.shape[0],)
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)

@keras_export('keras.backend.local_conv2d')
@dispatch.add_dispatch_support
def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format=None):
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)

@keras_export('keras.backend.bias_add')
@dispatch.add_dispatch_support
def bias_add(x, bias, data_format=None):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  bias_shape = int_shape(bias)
  if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
    raise ValueError(
        'Unexpected bias dimensions %d, expect to be 1 or %d dimensions' %
        (len(bias_shape), ndim(x)))

  if len(bias_shape) == 1:
    if data_format == 'channels_first':
      return nn.bias_add(x, bias, data_format='NCHW')
    return nn.bias_add(x, bias, data_format='NHWC')
  if ndim(x) in (3, 4, 5):
    if data_format == 'channels_first':
      bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
      return x + reshape(bias, bias_reshape_axis)
    return x + reshape(bias, (1,) + bias_shape)
  return nn.bias_add(x, bias)

@keras_export('keras.backend.random_normal')
@dispatch.add_dispatch_support
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_normal(
      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

@keras_export('keras.backend.random_uniform')
@dispatch.add_dispatch_support
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_uniform(
      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)

@deprecated(None, 'Use `tf.keras.backend.random_bernoulli` instead.')
@keras_export('keras.backend.random_binomial')
@dispatch.add_dispatch_support
def random_binomial(shape, p=0.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return array_ops.where_v2(
      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))

@keras_export('keras.backend.random_bernoulli')
@dispatch.add_dispatch_support
def random_bernoulli(shape, p=0.0, dtype=None, seed=None):
  return random_binomial(shape, p, dtype, seed)

@keras_export('keras.backend.truncated_normal')
@dispatch.add_dispatch_support
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.truncated_normal(
      shape, mean, stddev, dtype=dtype, seed=seed)

@keras_export('keras.backend.ctc_label_dense_to_sparse')
@dispatch.add_dispatch_support
def ctc_label_dense_to_sparse(labels, label_lengths):
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops.stack([label_shape[0]])
  max_num_labels_tns = array_ops.stack([label_shape[1]])

  def range_less_than(old_input, current_input):
    return array_ops.expand_dims(
        math_ops.range(array_ops.shape(old_input)[1]), 0) < array_ops.fill(
            max_num_labels_tns, current_input)

  init = math_ops.cast(
      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
  dense_mask = functional_ops.scan(
      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
  dense_mask = dense_mask[:, 0, :]

  label_array = array_ops.reshape(
      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
      label_shape)
  label_ind = array_ops.boolean_mask(label_array, dense_mask)

  batch_array = array_ops.transpose(
      array_ops.reshape(
          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
          reverse(label_shape, 0)))
  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
  indices = array_ops.transpose(
      array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

  vals_sparse = array_ops.gather_nd(labels, indices)

  return sparse_tensor.SparseTensor(
      math_ops.cast(indices, dtypes_module.int64), vals_sparse,
      math_ops.cast(label_shape, dtypes_module.int64))

@keras_export('keras.backend.ctc_batch_cost')
@dispatch.add_dispatch_support
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
  label_length = math_ops.cast(
      array_ops.squeeze(label_length, axis=-1), dtypes_module.int32)
  input_length = math_ops.cast(
      array_ops.squeeze(input_length, axis=-1), dtypes_module.int32)
  sparse_labels = math_ops.cast(
      ctc_label_dense_to_sparse(y_true, label_length), dtypes_module.int32)

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)

@keras_export('keras.backend.ctc_decode')
@dispatch.add_dispatch_support
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
  input_shape = shape(y_pred)
  num_samples, num_steps = input_shape[0], input_shape[1]
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
  input_length = math_ops.cast(input_length, dtypes_module.int32)

  if greedy:
    (decoded, log_prob) = ctc.ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length)
  else:
    (decoded, log_prob) = ctc.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths)
  decoded_dense = []
  for st in decoded:
    st = sparse_tensor.SparseTensor(
        st.indices, st.values, (num_samples, num_steps))
    decoded_dense.append(
        sparse_ops.sparse_tensor_to_dense(sp_input=st, default_value=-1))
  return (decoded_dense, log_prob)

@keras_export('keras.backend.map_fn')
def map_fn(fn, elems, name=None, dtype=None):
  return map_fn_lib.map_fn(fn, elems, name=name, dtype=dtype)

@keras_export('keras.backend.foldl')
def foldl(fn, elems, initializer=None, name=None):
  return functional_ops.foldl(fn, elems, initializer=initializer, name=name)

@keras_export('keras.backend.foldr')
def foldr(fn, elems, initializer=None, name=None):
  return functional_ops.foldr(fn, elems, initializer=initializer, name=name)

if 'KERAS_HOME' in os.environ:
  _keras_dir = os.environ.get('KERAS_HOME')
else:
  _keras_base_dir = os.path.expanduser('~')
  _keras_dir = os.path.join(_keras_base_dir, '.keras')
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
  try:
    with open(_config_path) as fh:
      _config = json.load(fh)
  except ValueError:
    _config = {}
  _floatx = _config.get('floatx', floatx())
  assert _floatx in {'float16', 'float32', 'float64'}
  _epsilon = _config.get('epsilon', epsilon())
  assert isinstance(_epsilon, float)
  _image_data_format = _config.get('image_data_format', image_data_format())
  assert _image_data_format in {'channels_last', 'channels_first'}
  set_floatx(_floatx)
  set_epsilon(_epsilon)
  set_image_data_format(_image_data_format)

if not os.path.exists(_keras_dir):
  try:
    os.makedirs(_keras_dir)
  except OSError:
    pass

if not os.path.exists(_config_path):
  _config = {
      'floatx': floatx(),
      'epsilon': epsilon(),
      'backend': 'tensorflow',
      'image_data_format': image_data_format()
  }
  try:
    with open(_config_path, 'w') as f:
      f.write(json.dumps(_config, indent=4))
  except IOError:
    pass

def configure_and_create_distributed_session(distribution_strategy):

  def _create_session(distribution_strategy):
    session_config = get_default_session_config()

    global _SESSION
    if getattr(_SESSION, 'session', None) and _SESSION.session._config:
      session_config.MergeFrom(_SESSION.session._config)

    if is_tpu_strategy(distribution_strategy):
      distribution_strategy.configure(session_config)
      master = distribution_strategy.extended._tpu_cluster_resolver.master()  
      session = session_module.Session(config=session_config, target=master)
    else:
      worker_context = dc_context.get_current_worker_context()
      if worker_context:
        dc_session_config = worker_context.session_config
        dc_session_config.MergeFrom(session_config)
        session = session_module.Session(
            config=dc_session_config, target=worker_context.master_target)
      else:
        distribution_strategy.configure(session_config)
        session = session_module.Session(config=session_config)

    set_session(session)

  if distribution_strategy.extended._in_multi_worker_mode():
    dc.run_distribute_coordinator(
        _create_session,
        distribution_strategy,
        mode=dc.CoordinatorMode.INDEPENDENT_WORKER)
  else:
    _create_session(distribution_strategy)

def is_tpu_strategy(strategy):
  return (strategy is not None and
          strategy.__class__.__name__.startswith('TPUStrategy'))

def cast_variables_to_tensor(tensors):

  def _cast_variables_to_tensor(tensor):
    if isinstance(tensor, variables_module.Variable):
      return array_ops.identity(tensor)
    return tensor

  return nest.map_structure(_cast_variables_to_tensor, tensors)

def _is_symbolic_tensor(x):
  return tensor_util.is_tensor(x) and not isinstance(x, ops.EagerTensor)

def convert_inputs_if_ragged(inputs):

  def _convert_ragged_input(inputs):
    if isinstance(inputs, ragged_tensor.RaggedTensor):
      return inputs.to_tensor()
    return inputs

  flat_inputs = nest.flatten(inputs)
  contains_ragged = py_any(
      isinstance(i, ragged_tensor.RaggedTensor) for i in flat_inputs)

  if not contains_ragged:
    return inputs, None

  inputs = nest.map_structure(_convert_ragged_input, inputs)
  nested_row_lengths = math_ops.cast(flat_inputs[0].nested_row_lengths()[0],
                                     'int32')
  return inputs, nested_row_lengths

def maybe_convert_to_ragged(is_ragged_input, output, nested_row_lengths):
  if not is_ragged_input:
    return output

  return ragged_tensor.RaggedTensor.from_tensor(output, nested_row_lengths)

class ContextValueCache(weakref.WeakKeyDictionary):

  def __init__(self, default_factory):
    self.default_factory = default_factory
    weakref.WeakKeyDictionary.__init__(self)

  def _key(self):
    if context.executing_eagerly():
      return _DUMMY_EAGER_GRAPH.key
    else:
      return ops.get_default_graph()

  def _get_parent_graph(self, graph):
    parent_graph = graph.outer_graph
    if (not isinstance(parent_graph, func_graph.FuncGraph) and
        ops.executing_eagerly_outside_functions()):
      return _DUMMY_EAGER_GRAPH.key
    return parent_graph

  def _get_recursive(self, key):
    value = self.get(key)
    if value is not None:
      return value

    if isinstance(key, func_graph.FuncGraph):
      return self._get_recursive(self._get_parent_graph(key))
    return None

  def __getitem__(self, key):
    if key is None:
      key = self._key()

    value = self._get_recursive(key)
    if value is None:
      value = self[key] = self.default_factory()  
    return value

  def setdefault(self, key=None, default=None, kwargs=None):
    if key is None:
      key = self._key()
    kwargs = kwargs or {}

    if default is None and key not in self:
      default = self.default_factory(**kwargs)
    return weakref.WeakKeyDictionary.setdefault(self, key, default)

_GRAPH_LEARNING_PHASES = ContextValueCache(_default_learning_phase)

_GRAPH_VARIABLES = ContextValueCache(object_identity.ObjectIdentityWeakSet)

_GRAPH_TF_OPTIMIZERS = ContextValueCache(object_identity.ObjectIdentityWeakSet)
EOF
