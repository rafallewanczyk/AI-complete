

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
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tfdev
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
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
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

py_all = all
py_sum = sum

_GRAPH = None

_CURRENT_SCRATCH_GRAPH = None

_SESSION = threading.local()

_GRAPH_LEARNING_PHASES = weakref.WeakKeyDictionary()

_DUMMY_EAGER_GRAPH = threading.local()

_MANUAL_VAR_INIT = False

_LOCAL_DEVICES = None

_GRAPH_VARIABLES = weakref.WeakKeyDictionary()

_GRAPH_TF_OPTIMIZERS = weakref.WeakKeyDictionary()

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
def cast_to_floatx(x):
  return np.asarray(x, dtype=floatx())

PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()

@keras_export('keras.backend.get_uid')
def get_uid(prefix=''):
  graph = get_graph()
  if graph not in PER_GRAPH_LAYER_NAME_UIDS:
    PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]

@keras_export('keras.backend.reset_uids')
def reset_uids():
  per_graph_layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS
  keys = list(per_graph_layer_name_uids.keys())
  for key in keys:
    del per_graph_layer_name_uids[key]

@keras_export('keras.backend.clear_session')
def clear_session():
  global _SESSION
  global _GRAPH_LEARNING_PHASES  
  global _GRAPH_VARIABLES  
  global _GRAPH_TF_OPTIMIZERS  
  ops.reset_default_graph()
  reset_uids()
  _SESSION.session = None
  graph = get_graph()
  with graph.as_default():
    with ops.name_scope(''):
      phase = array_ops.placeholder_with_default(
          False, shape=(), name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES = {}
    _GRAPH_LEARNING_PHASES[graph] = phase
    _GRAPH_VARIABLES.pop(graph, None)
    _GRAPH_TF_OPTIMIZERS.pop(graph, None)

@keras_export('keras.backend.manual_variable_initialization')
def manual_variable_initialization(value):
  global _MANUAL_VAR_INIT
  _MANUAL_VAR_INIT = value

@keras_export('keras.backend.learning_phase')
def learning_phase():
  if ops.get_default_graph() is _GRAPH:
    return symbolic_learning_phase()
  with ops.init_scope():
    if context.executing_eagerly():
      if _DUMMY_EAGER_GRAPH not in _GRAPH_LEARNING_PHASES:
        return 0
      return _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH]
    return symbolic_learning_phase()

def symbolic_learning_phase():
  graph = get_graph()
  with graph.as_default():
    if graph not in _GRAPH_LEARNING_PHASES:
      with ops.name_scope(''):
        phase = array_ops.placeholder_with_default(
            False, shape=(), name='keras_learning_phase')
      _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]

@keras_export('keras.backend.set_learning_phase')
def set_learning_phase(value):
  global _GRAPH_LEARNING_PHASES  
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  with ops.init_scope():
    if context.executing_eagerly():
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value
    _GRAPH_LEARNING_PHASES[get_graph()] = value

def set_eager_learning_phase(value):
  global _GRAPH_LEARNING_PHASES  
  assert value in {0, 1}
  assert context.executing_eagerly()
  _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value

@keras_export('keras.backend.learning_phase_scope')
@tf_contextlib.contextmanager
def learning_phase_scope(value):
  global _GRAPH_LEARNING_PHASES  
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')

  with ops.init_scope():
    if context.executing_eagerly():
      previous_eager_value = _GRAPH_LEARNING_PHASES.get(
          _DUMMY_EAGER_GRAPH, None)
    previous_graph_value = _GRAPH_LEARNING_PHASES.get(get_graph(), None)

  try:
    set_learning_phase(value)
    yield
  finally:
    with ops.init_scope():
      if context.executing_eagerly():
        if previous_eager_value is not None:
          _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = previous_eager_value
        elif _DUMMY_EAGER_GRAPH in _GRAPH_LEARNING_PHASES:
          del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH]

      graph = get_graph()
      if previous_graph_value is not None:
        _GRAPH_LEARNING_PHASES[graph] = previous_graph_value
      elif graph in _GRAPH_LEARNING_PHASES:
        del _GRAPH_LEARNING_PHASES[graph]

@tf_contextlib.contextmanager
def eager_learning_phase_scope(value):
  global _GRAPH_LEARNING_PHASES  
  assert value in {0, 1}
  assert context.executing_eagerly()
  previous_value = learning_phase()
  try:
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value
    yield
  finally:
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = previous_value

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

def get_graph():
  if context.executing_eagerly():
    global _GRAPH
    if _GRAPH is None:
      _GRAPH = func_graph.FuncGraph('keras_graph')
    return _GRAPH
  else:
    return ops.get_default_graph()

@tf_contextlib.contextmanager
def _scratch_graph(graph=None):
  global _CURRENT_SCRATCH_GRAPH
  if (_CURRENT_SCRATCH_GRAPH is not None and graph is not None and
      _CURRENT_SCRATCH_GRAPH is not graph):
    raise ValueError('Multiple scratch graphs specified.')

  if _CURRENT_SCRATCH_GRAPH:
    yield _CURRENT_SCRATCH_GRAPH
    return

  graph = graph or func_graph.FuncGraph('keras_scratch_graph')
  try:
    _CURRENT_SCRATCH_GRAPH = graph
    yield graph
  finally:
    _CURRENT_SCRATCH_GRAPH = None

@keras_export('keras.backend.set_session')
def set_session(session):
  global _SESSION
  _SESSION.session = session

def get_default_session_config():
  if not os.environ.get('OMP_NUM_THREADS'):
    config = config_pb2.ConfigProto(allow_soft_placement=True)
  else:
    num_thread = int(os.environ.get('OMP_NUM_THREADS'))
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=num_thread,
        inter_op_parallelism_threads=num_thread,
        allow_soft_placement=True)
  return config

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
    return [name for name in context.list_devices() if 'GPU' in name]

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
  return ops.convert_to_tensor(x, dtype=dtype)

@keras_export('keras.backend.is_sparse')
def is_sparse(tensor):
  return isinstance(tensor, sparse_tensor.SparseTensor)

@keras_export('keras.backend.to_dense')
def to_dense(tensor):
  if is_sparse(tensor):
    return sparse_ops.sparse_tensor_to_dense(tensor)
  else:
    return tensor

name_scope = ops.name_scope

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
  v = resource_variable_ops.ResourceVariable(
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
  graph = get_graph()
  optimizers = _GRAPH_TF_OPTIMIZERS.setdefault(graph, weakref.WeakSet())
  optimizers.add(tf_optimizer)

def track_variable(v):
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else get_graph()
  if graph not in _GRAPH_VARIABLES:
    _GRAPH_VARIABLES[graph] = weakref.WeakSet()
  _GRAPH_VARIABLES[graph].add(v)

def _get_variables(graph=None):
  assert not context.executing_eagerly()
  variables = _GRAPH_VARIABLES.setdefault(graph, weakref.WeakSet())
  for opt in _GRAPH_TF_OPTIMIZERS.get(graph, set()):
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
    uninitialized_vars = []
    for flag, v in zip(is_initialized, candidate_vars):
      if not flag:
        uninitialized_vars.append(v)
      v._keras_initialized = True
    if uninitialized_vars:
      session.run(variables_module.variables_initializer(uninitialized_vars))

@keras_export('keras.backend.constant')
def constant(value, dtype=None, shape=None, name=None):
  if dtype is None:
    dtype = floatx()

  if (ops.executing_eagerly_outside_functions() and
      getattr(get_graph(), 'name', '') == 'keras_graph'):
    with ops.init_scope():
      return constant_op.constant(value, dtype=dtype, shape=shape, name=name)

  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)

def is_keras_tensor(x):
  if not isinstance(x, (ops.Tensor,
                        variables_module.Variable,
                        sparse_tensor.SparseTensor)):
    raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                     '`. Expected a symbolic tensor instance.')
  return hasattr(x, '_keras_history')

@keras_export('keras.backend.placeholder')
def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
  if dtype is None:
    dtype = floatx()
  if not shape:
    if ndim:
      shape = tuple([None for _ in range(ndim)])
  with get_graph().as_default():
    if sparse:
      x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
    else:
      x = array_ops.placeholder(dtype, shape=shape, name=name)
  return x

def is_placeholder(x):
  try:
    return x.op.type == 'Placeholder'
  except AttributeError:
    return False

@keras_export('keras.backend.shape')
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
    track_variable(v)
    return v

@keras_export('keras.backend.ones')
def ones(shape, dtype=None, name=None):
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.ones(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    track_variable(v)
    return v

@keras_export('keras.backend.eye')
def eye(size, dtype=None, name=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  return variable(linalg_ops.eye(size, dtype=tf_dtype), dtype, name)

@keras_export('keras.backend.zeros_like')
def zeros_like(x, dtype=None, name=None):
  return array_ops.zeros_like(x, dtype=dtype, name=name)

@keras_export('keras.backend.ones_like')
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
  from tensorflow.python.training import moving_averages  
  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=True)

@keras_export('keras.backend.dot')
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
def batch_dot(x, y, axes=None):
  if isinstance(axes, int):
    axes = (axes, axes)
  x_ndim = ndim(x)
  y_ndim = ndim(y)
  if axes is None:
    axes = [x_ndim - 1, y_ndim - 2]
  if x_ndim > y_ndim:
    diff = x_ndim - y_ndim
    y = array_ops.reshape(y,
                          array_ops.concat(
                              [array_ops.shape(y), [1] * (diff)], axis=0))
  elif y_ndim > x_ndim:
    diff = y_ndim - x_ndim
    x = array_ops.reshape(x,
                          array_ops.concat(
                              [array_ops.shape(x), [1] * (diff)], axis=0))
  else:
    diff = 0
  if ndim(x) == 2 and ndim(y) == 2:
    if axes[0] == axes[1]:
      out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
    else:
      out = math_ops.reduce_sum(
          math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
  else:
    adj_x = None if axes[0] == ndim(x) - 1 else True
    adj_y = True if axes[1] == ndim(y) - 1 else None
    out = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
  if diff:
    if x_ndim > y_ndim:
      idx = x_ndim + y_ndim - 3
    else:
      idx = x_ndim - 1
    out = array_ops.squeeze(out, list(range(idx, idx + diff)))
  if ndim(out) == 1:
    out = expand_dims(out, 1)
  return out

@keras_export('keras.backend.transpose')
def transpose(x):
  return array_ops.transpose(x)

@keras_export('keras.backend.gather')
def gather(reference, indices):
  return array_ops.gather(reference, indices)

@keras_export('keras.backend.max')
def max(x, axis=None, keepdims=False):
  return math_ops.reduce_max(x, axis, keepdims)

@keras_export('keras.backend.min')
def min(x, axis=None, keepdims=False):
  return math_ops.reduce_min(x, axis, keepdims)

@keras_export('keras.backend.sum')
def sum(x, axis=None, keepdims=False):
  return math_ops.reduce_sum(x, axis, keepdims)

@keras_export('keras.backend.prod')
def prod(x, axis=None, keepdims=False):
  return math_ops.reduce_prod(x, axis, keepdims)

@keras_export('keras.backend.cumsum')
def cumsum(x, axis=0):
  return math_ops.cumsum(x, axis=axis)

@keras_export('keras.backend.cumprod')
def cumprod(x, axis=0):
  return math_ops.cumprod(x, axis=axis)

@keras_export('keras.backend.var')
def var(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_variance(x, axis=axis, keepdims=keepdims)

@keras_export('keras.backend.std')
def std(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_std(x, axis=axis, keepdims=keepdims)

@keras_export('keras.backend.mean')
def mean(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, axis, keepdims)

@keras_export('keras.backend.any')
def any(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, axis, keepdims)

@keras_export('keras.backend.all')
def all(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, axis, keepdims)

@keras_export('keras.backend.argmax')
def argmax(x, axis=-1):
  return math_ops.argmax(x, axis)

@keras_export('keras.backend.argmin')
def argmin(x, axis=-1):
  return math_ops.argmin(x, axis)

@keras_export('keras.backend.square')
def square(x):
  return math_ops.square(x)

@keras_export('keras.backend.abs')
def abs(x):
  return math_ops.abs(x)

@keras_export('keras.backend.sqrt')
def sqrt(x):
  zero = _constant_to_tensor(0., x.dtype.base_dtype)
  inf = _constant_to_tensor(np.inf, x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, inf)
  return math_ops.sqrt(x)

@keras_export('keras.backend.exp')
def exp(x):
  return math_ops.exp(x)

@keras_export('keras.backend.log')
def log(x):
  return math_ops.log(x)

def logsumexp(x, axis=None, keepdims=False):
  return math_ops.reduce_logsumexp(x, axis, keepdims)

@keras_export('keras.backend.round')
def round(x):
  return math_ops.round(x)

@keras_export('keras.backend.sign')
def sign(x):
  return math_ops.sign(x)

@keras_export('keras.backend.pow')
def pow(x, a):
  return math_ops.pow(x, a)

@keras_export('keras.backend.clip')
def clip(x, min_value, max_value):
  if max_value is not None and max_value < min_value:
    max_value = min_value
  if max_value is None:
    max_value = np.inf
  min_value = _constant_to_tensor(min_value, x.dtype.base_dtype)
  max_value = _constant_to_tensor(max_value, x.dtype.base_dtype)
  return clip_ops.clip_by_value(x, min_value, max_value)

@keras_export('keras.backend.equal')
def equal(x, y):
  return math_ops.equal(x, y)

@keras_export('keras.backend.not_equal')
def not_equal(x, y):
  return math_ops.not_equal(x, y)

@keras_export('keras.backend.greater')
def greater(x, y):
  return math_ops.greater(x, y)

@keras_export('keras.backend.greater_equal')
def greater_equal(x, y):
  return math_ops.greater_equal(x, y)

@keras_export('keras.backend.less')
def less(x, y):
  return math_ops.less(x, y)

@keras_export('keras.backend.less_equal')
def less_equal(x, y):
  return math_ops.less_equal(x, y)

@keras_export('keras.backend.maximum')
def maximum(x, y):
  return math_ops.maximum(x, y)

@keras_export('keras.backend.minimum')
def minimum(x, y):
  return math_ops.minimum(x, y)

@keras_export('keras.backend.sin')
def sin(x):
  return math_ops.sin(x)

@keras_export('keras.backend.cos')
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
def concatenate(tensors, axis=-1):
  if axis < 0:
    rank = ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if py_all(is_sparse(x) for x in tensors):
    return sparse_ops.sparse_concat(axis, tensors)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)

@keras_export('keras.backend.reshape')
def reshape(x, shape):
  return array_ops.reshape(x, shape)

@keras_export('keras.backend.permute_dimensions')
def permute_dimensions(x, pattern):
  return array_ops.transpose(x, perm=pattern)

@keras_export('keras.backend.resize_images')
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
    x = image_ops.resize_nearest_neighbor(x, new_shape)
  elif interpolation == 'bilinear':
    x = image_ops.resize_bilinear(x, new_shape)
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
def repeat(x, n):
  assert ndim(x) == 2
  x = array_ops.expand_dims(x, 1)
  pattern = array_ops.stack([1, n, 1])
  return array_ops.tile(x, pattern)

@keras_export('keras.backend.arange')
def arange(start, stop=None, step=1, dtype='int32'):
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result

@keras_export('keras.backend.tile')
def tile(x, n):
  if isinstance(n, int):
    n = [n]
  return array_ops.tile(x, n)

@keras_export('keras.backend.flatten')
def flatten(x):
  return array_ops.reshape(x, [-1])

@keras_export('keras.backend.batch_flatten')
def batch_flatten(x):
  x = array_ops.reshape(x, array_ops.stack([-1, prod(shape(x)[1:])]))
  return x

@keras_export('keras.backend.expand_dims')
def expand_dims(x, axis=-1):
  return array_ops.expand_dims(x, axis)

@keras_export('keras.backend.squeeze')
def squeeze(x, axis):
  return array_ops.squeeze(x, [axis])

@keras_export('keras.backend.temporal_padding')
def temporal_padding(x, padding=(1, 1)):
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return array_ops.pad(x, pattern)

@keras_export('keras.backend.spatial_2d_padding')
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
def stack(x, axis=0):
  return array_ops.stack(x, axis=axis)

@keras_export('keras.backend.one_hot')
def one_hot(indices, num_classes):
  return array_ops.one_hot(indices, depth=num_classes, axis=-1)

@keras_export('keras.backend.reverse')
def reverse(x, axes):
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)

@keras_export('keras.backend.get_value')
def get_value(x):
  if not tensor_util.is_tensor(x):
    return x
  if context.executing_eagerly():
    return x.numpy()
  if not getattr(x, '_in_graph_mode', True):
    with context.eager_mode():
      return x.numpy()

  if ops.executing_eagerly_outside_functions():
    return function([], x)(x)

  return x.eval(session=get_session((x,)))

@keras_export('keras.backend.batch_get_value')
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
        assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      get_session().run(assign_op, feed_dict={assign_placeholder: value})

@keras_export('keras.backend.batch_set_value')
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
            assign_placeholder = array_ops.placeholder(tf_dtype,
                                                       shape=value.shape)
            assign_op = x.assign(assign_placeholder)
            x._assign_placeholder = assign_placeholder
            x._assign_op = assign_op
          assign_ops.append(assign_op)
          feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)

@keras_export('keras.backend.print_tensor')
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
    self.inputs = nest.flatten(inputs)
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
    inputs = nest.flatten(inputs)

    session = get_session(inputs)
    feed_arrays = []
    array_vals = []
    feed_symbols = []
    symbol_vals = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        continue
      if is_sparse(tensor):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        value = (indices, sparse_coo.data, sparse_coo.shape)
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

class EagerExecutionFunction(object):

  def __init__(self, inputs, outputs, updates=None, name=None):
    self.name = name
    self._outputs_structure = outputs
    inputs = nest.flatten(inputs)
    outputs = nest.flatten(outputs, expand_composites=True)

    updates = updates or []
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a Keras backend function '
                      'should be a list or tuple.')

    if updates and not outputs:
      raise ValueError('Cannot create a Keras backend function with updates'
                       ' but no outputs during eager execution.')

    graphs = {i.graph for i in nest.flatten([inputs, outputs, updates])
              if hasattr(i, 'graph')}
    if len(graphs) > 1:
      raise ValueError('Cannot create an execution function which is comprised '
                       'of elements from multiple graphs.')

    source_graph = graphs.pop()
    global_graph = get_graph()

    updates_ops = []
    legacy_update_ops = []
    for update in updates:
      if isinstance(update, tuple):
        legacy_update_ops.append(update)
      else:
        if hasattr(update, 'op'):
          update = update.op
        if update is not None:
          updates_ops.append(update)

    with _scratch_graph() as exec_graph:
      global_graph = get_graph()
      if source_graph not in (exec_graph, global_graph):
        raise ValueError('Unknown graph. Aborting.')

      if source_graph is global_graph and exec_graph is not global_graph:
        init_tensors = (
            outputs + updates_ops + [p for [p, _] in legacy_update_ops] +
            [p_new for [_, p_new] in legacy_update_ops
             if isinstance(p_new, ops.Tensor)])
        lifted_map = lift_to_graph.lift_to_graph(
            init_tensors=init_tensors, graph=exec_graph, sources=inputs,
            add_sources=True, handle_captures=True, base_graph=source_graph)

        inputs = [lifted_map[i] for i in inputs]
        outputs = [lifted_map[i] for i in outputs]
        updates_ops = [lifted_map[i] for i in updates_ops]
        legacy_update_ops = [(lifted_map[p], lifted_map.get(p_new, p_new))
                             for p, p_new in legacy_update_ops]

    with exec_graph.as_default():
      outputs = cast_variables_to_tensor(outputs)
      with ops.control_dependencies(outputs):
        for p, p_new in legacy_update_ops:
          updates_ops.append(state_ops.assign(p, p_new))

      self.inputs, self.outputs = inputs, outputs
      with ops.control_dependencies(updates_ops):
        self.outputs[0] = array_ops.identity(self.outputs[0])

      exec_graph.inputs = self.inputs + list(exec_graph.captures.values())
      exec_graph.outputs = self.outputs
      graph_fn = eager_function.ConcreteFunction(exec_graph)

    graph_fn._num_positional_args = len(self.inputs)
    graph_fn._arg_keywords = []
    self._graph_fn = graph_fn

    self._placeholder_default_values = {}
    with exec_graph.as_default():
      for x in self.inputs:
        if x.op.type == 'PlaceholderWithDefault':
          self._placeholder_default_values[x] = tensor_util.constant_value(
              x.op.inputs[0])

  def __call__(self, inputs):
    inputs = nest.flatten(inputs)
    converted_inputs = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        value = self._placeholder_default_values.get(tensor, None)
        if value is None:
          raise ValueError(
              'You must feed a value for placeholder %s' % (tensor,))
      if not isinstance(value, ops.Tensor):
        value = ops.convert_to_tensor(value, dtype=tensor.dtype)
      if value.dtype != tensor.dtype:
        value = math_ops.cast(value, tensor.dtype)
      converted_inputs.append(value)
    outputs = self._graph_fn(*converted_inputs)
    return nest.pack_sequence_as(
        self._outputs_structure, [x.numpy() for x in outputs],
        expand_composites=True)

@keras_export('keras.backend.function')
def function(inputs, outputs, updates=None, name=None, **kwargs):
  if ops.executing_eagerly_outside_functions():
    if kwargs:
      raise ValueError('Session keyword arguments are not support during '
                       'eager execution. You passed: %s' % (kwargs,))
    return EagerExecutionFunction(inputs, outputs, updates=updates, name=name)

  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in ['inputs', 'outputs', 'updates', 'name']):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return GraphExecutionFunction(inputs, outputs, updates=updates, **kwargs)

@keras_export('keras.backend.gradients')
def gradients(loss, variables):
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)

@keras_export('keras.backend.stop_gradient')
def stop_gradient(variables):
  if isinstance(variables, (list, tuple)):
    return map(array_ops.stop_gradient, variables)
  return array_ops.stop_gradient(variables)

@keras_export('keras.backend.rnn')
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
    assert not nest.is_sequence(mask_t)
    assert not nest.is_sequence(input_t)
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

        output = array_ops.where(tiled_mask_t, output, prev_output)

        return_states = []
        for state, new_state in zip(states, new_states):
          tiled_mask_t = _expand_mask(mask_t, new_state)
          return_states.append(array_ops.where(tiled_mask_t, new_state, state))
        states = return_states
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where(
            _expand_mask(mask_list[-1], last_output),
            last_output,
            zeros_like(last_output))
        outputs = array_ops.where(
            _expand_mask(mask, outputs, fixed_dim=2),
            outputs,
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
    output_time_zero, _ = step_function(input_time_zero,
                                        initial_states + constants)
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': input_length,
        'parallel_iterations': 32,
        'swap_memory': True,
    }

    if mask is not None:
      if not states:
        raise ValueError('No initial states provided! '
                         'When using masking in an RNN, you should '
                         'provide initial states '
                         '(and your step function should return '
                         'as its first state at time `t` '
                         'the output at time `t-1`).')
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time, output_ta_t, prev_output, *states):
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = mask_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask
                            else nest.flatten(prev_output))
        tiled_mask_t = tuple(_expand_mask(mask_t, o) for o in flat_output)
        flat_new_output = tuple(
            array_ops.where(m, o, zo) for m, o, zo in zip(
                tiled_mask_t, flat_output, flat_mask_output))

        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, ops.Tensor):
            new_state.set_shape(state.shape)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_state)
        flat_final_state = tuple(
            array_ops.where(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_state, flat_state))
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
      tile_shape = array_ops.where(shape_diff > 0, expr_shape,
                                   array_ops.ones_like(expr_shape))
      condition = array_ops.tile(condition, tile_shape)
    x = array_ops.where(condition, then_expression, else_expression)
  return x

@keras_export('keras.backend.in_train_phase')
def in_train_phase(x, alt, training=None):
  if training is None:
    training = learning_phase()

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
def relu(x, alpha=0., max_value=None, threshold=0):

  if alpha != 0.:
    if max_value is None and threshold == 0:
      return nn.leaky_relu(x, alpha=alpha)

    if threshold != 0:
      negative_part = nn.relu(-x + threshold)
    else:
      negative_part = nn.relu(-x)

  clip_max = max_value is not None

  if threshold != 0:
    x = x * math_ops.cast(math_ops.greater(x, threshold), floatx())
  elif max_value == 6:
    x = nn.relu6(x)
    clip_max = False
  else:
    x = nn.relu(x)

  if clip_max:
    max_value = _constant_to_tensor(max_value, x.dtype.base_dtype)
    zero = _constant_to_tensor(0., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x

@keras_export('keras.backend.elu')
def elu(x, alpha=1.):
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where(x > 0, res, alpha * res)

@keras_export('keras.backend.softmax')
def softmax(x, axis=-1):
  return nn.softmax(x, axis=axis)

@keras_export('keras.backend.softplus')
def softplus(x):
  return nn.softplus(x)

@keras_export('keras.backend.softsign')
def softsign(x):
  return nn.softsign(x)

@keras_export('keras.backend.categorical_crossentropy')
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  if not from_logits:
    if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
        output.op.type != 'Softmax'):
      axis = axis % len(output.shape)
      output = output / math_ops.reduce_sum(output, axis, True)
      epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
      return -math_ops.reduce_sum(target * math_ops.log(output), axis)
    else:
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]
  return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)

@keras_export('keras.backend.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  if not from_logits:
    if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
        output.op.type != 'Softmax'):
      epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
      output = math_ops.log(output)
    else:
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]

  rank = len(output.shape)
  axis = axis % rank
  if axis != rank - 1:
    permutation = list(range(axis)) + list(range(axis + 1, rank)) + [axis]
    output = array_ops.transpose(output, perm=permutation)

  output_shape = output.shape
  targets = cast(flatten(target), 'int64')
  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
  res = nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
  if len(output_shape) >= 3:
    return array_ops.reshape(res, array_ops.shape(output)[:-1])
  else:
    return res

@keras_export('keras.backend.binary_crossentropy')
def binary_crossentropy(target, output, from_logits=False):
  if not from_logits:
    if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
        output.op.type != 'Sigmoid'):
      epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

      bce = target * math_ops.log(output + epsilon())
      bce += (1 - target) * math_ops.log(1 - output + epsilon())
      return -bce
    else:
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]
  return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

@keras_export('keras.backend.sigmoid')
def sigmoid(x):
  return nn.sigmoid(x)

@keras_export('keras.backend.hard_sigmoid')
def hard_sigmoid(x):
  point_two = _constant_to_tensor(0.2, x.dtype.base_dtype)
  point_five = _constant_to_tensor(0.5, x.dtype.base_dtype)
  x = math_ops.mul(x, point_two)
  x = math_ops.add(x, point_five)
  x = clip_ops.clip_by_value(x, 0., 1.)
  return x

@keras_export('keras.backend.tanh')
def tanh(x):
  return nn.tanh(x)

@keras_export('keras.backend.dropout')
def dropout(x, level, noise_shape=None, seed=None):
  if seed is None:
    seed = np.random.randint(10e6)
  return nn.dropout_v2(x, rate=level, noise_shape=noise_shape, seed=seed)

@keras_export('keras.backend.l2_normalize')
def l2_normalize(x, axis=None):
  return nn.l2_normalize(x, axis=axis)

@keras_export('keras.backend.in_top_k')
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

    slices.extend([slice(position[d] * strides[d],
                         position[d] * strides[d] + kernel_size[d])
                   for d in spatial_dimensions])

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
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  output_shape = (kernel.shape[0],)
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)

@keras_export('keras.backend.local_conv2d')
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
  if ndim(x) == 5:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, bias_shape[0], 1, 1, 1))
      else:
        x = x + reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, 1, 1, bias_shape[0]))
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 4:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        if _has_nchw_support():
          x = nn.bias_add(x, bias, data_format='NCHW')
        else:
          x = x + reshape(bias, (1, bias_shape[0], 1, 1))
      else:
        x = x + reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = nn.bias_add(x, bias, data_format='NHWC')
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 3:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, bias_shape[0], 1))
      else:
        x = x + reshape(bias, (1, bias_shape[1], bias_shape[0]))
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, 1, bias_shape[0]))
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  else:
    x = nn.bias_add(x, bias)
  return x

@keras_export('keras.backend.random_normal')
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_normal(
      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

@keras_export('keras.backend.random_uniform')
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_uniform(
      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)

@keras_export('keras.backend.random_binomial')
def random_binomial(shape, p=0.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return array_ops.where(
      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))

@keras_export('keras.backend.truncated_normal')
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.truncated_normal(
      shape, mean, stddev, dtype=dtype, seed=seed)

@keras_export('keras.backend.ctc_label_dense_to_sparse')
def ctc_label_dense_to_sparse(labels, label_lengths):
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops.stack([label_shape[0]])
  max_num_labels_tns = array_ops.stack([label_shape[1]])

  def range_less_than(_, current_input):
    return array_ops.expand_dims(
        math_ops.range(label_shape[1]), 0) < array_ops.fill(
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
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
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
  decoded_dense = [
      sparse_ops.sparse_to_dense(
          st.indices, st.dense_shape, st.values, default_value=-1)
      for st in decoded
  ]
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
    _config = json.load(open(_config_path))
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

def in_multi_worker_mode():
  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_spec = server_lib.ClusterSpec(tf_config.get('cluster', {}))
  return tf_config and 'master' not in cluster_spec.jobs

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

  if in_multi_worker_mode():
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
EOF
