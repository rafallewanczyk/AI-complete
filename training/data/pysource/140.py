

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import os
import weakref

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_module
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

py_all = all
py_sum = sum

_SESSION = None

_GRAPH_LEARNING_PHASES = weakref.WeakKeyDictionary()

class _DummyEagerGraph(object):
  pass
_DUMMY_EAGER_GRAPH = _DummyEagerGraph()

_MANUAL_VAR_INIT = False

_FLOATX = 'float32'

_EPSILON = 1e-7

_IMAGE_DATA_FORMAT = 'channels_last'

_LOCAL_DEVICES = None

_GRAPH_VARIABLES = weakref.WeakKeyDictionary()

_GRAPH_TF_OPTIMIZERS = weakref.WeakKeyDictionary()

@tf_export('keras.backend.backend')
def backend():
  return 'tensorflow'

@tf_export('keras.backend.epsilon')
def epsilon():
  return _EPSILON

@tf_export('keras.backend.set_epsilon')
def set_epsilon(value):
  global _EPSILON
  _EPSILON = value

@tf_export('keras.backend.floatx')
def floatx():
  return _FLOATX

@tf_export('keras.backend.set_floatx')
def set_floatx(value):
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)

@tf_export('keras.backend.cast_to_floatx')
def cast_to_floatx(x):
  return np.asarray(x, dtype=_FLOATX)

@tf_export('keras.backend.image_data_format')
def image_data_format():
  return _IMAGE_DATA_FORMAT

@tf_export('keras.backend.set_image_data_format')
def set_image_data_format(data_format):
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  _IMAGE_DATA_FORMAT = str(data_format)

PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()

@tf_export('keras.backend.get_uid')
def get_uid(prefix=''):
  graph = ops.get_default_graph()
  if graph not in PER_GRAPH_LAYER_NAME_UIDS:
    PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]

@tf_export('keras.backend.reset_uids')
def reset_uids():
  per_graph_layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS
  keys = list(per_graph_layer_name_uids.keys())
  for key in keys:
    del per_graph_layer_name_uids[key]

@tf_export('keras.backend.clear_session')
def clear_session():
  global _SESSION
  global _GRAPH_LEARNING_PHASES  
  global _GRAPH_VARIABLES  
  global _GRAPH_TF_OPTIMIZERS  
  ops.reset_default_graph()
  reset_uids()
  _SESSION = None
  phase = array_ops.placeholder_with_default(
      False, shape=(), name='keras_learning_phase')
  _GRAPH_LEARNING_PHASES = {}
  _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = phase
  _GRAPH_VARIABLES.pop(ops.get_default_graph(), None)
  _GRAPH_TF_OPTIMIZERS.pop(ops.get_default_graph(), None)

@tf_export('keras.backend.manual_variable_initialization')
def manual_variable_initialization(value):
  global _MANUAL_VAR_INIT
  _MANUAL_VAR_INIT = value

@tf_export('keras.backend.learning_phase')
def learning_phase():
  with ops.init_scope():
    if context.executing_eagerly():
      if _DUMMY_EAGER_GRAPH not in _GRAPH_LEARNING_PHASES:
        return 0
      return _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH]

    graph = ops.get_default_graph()
    if graph not in _GRAPH_LEARNING_PHASES:
      phase = array_ops.placeholder_with_default(
          False, shape=(), name='keras_learning_phase')
      _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]

@tf_export('keras.backend.set_learning_phase')
def set_learning_phase(value):
  global _GRAPH_LEARNING_PHASES  
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  with ops.init_scope():
    if context.executing_eagerly():
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value
    else:
      _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = value

@tf_contextlib.contextmanager
def learning_phase_scope(value):
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  previous_value = learning_phase()
  try:
    set_learning_phase(value)
    yield value
  finally:
    with ops.init_scope():
      if context.executing_eagerly():
        _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = previous_value
      else:
        _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = previous_value

@tf_export('keras.backend.get_session')
def get_session():
  global _SESSION
  default_session = ops.get_default_session()
  if default_session is not None:
    session = default_session
  else:
    if _SESSION is None:
      _SESSION = session_module.Session(config=get_default_session_config())
    session = _SESSION
  if not _MANUAL_VAR_INIT:
    with session.graph.as_default():
      _initialize_variables(session)
  return session

@tf_export('keras.backend.set_session')
def set_session(session):
  global _SESSION
  _SESSION = session

def get_default_session_config():
  if not os.environ.get('OMP_NUM_THREADS'):
    config = config_pb2.ConfigProto(allow_soft_placement=True)
  else:
    num_thread = int(os.environ.get('OMP_NUM_THREADS'))
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=num_thread, allow_soft_placement=True)
  return config

class _TfDeviceCaptureOp(object):

  def __init__(self):
    self.device = None

  def _set_device(self, device):
    self.device = device

def _get_current_tf_device():
  g = ops.get_default_graph()
  op = _TfDeviceCaptureOp()
  g._apply_device_functions(op)
  return op.device

def _is_current_explicit_device(device_type):
  device_type = device_type.upper()
  if device_type not in ['CPU', 'GPU']:
    raise ValueError('`device_type` should be either "CPU" or "GPU".')
  device = _get_current_tf_device()
  return device is not None and device.device_type == device_type.upper()

def _get_available_gpus():
  global _LOCAL_DEVICES
  if _LOCAL_DEVICES is None:
    _LOCAL_DEVICES = get_session().list_devices()
  return [x.name for x in _LOCAL_DEVICES if x.device_type == 'GPU']

def _has_nchw_support():
  explicitly_on_cpu = _is_current_explicit_device('CPU')
  gpus_available = bool(_get_available_gpus())
  return not explicitly_on_cpu and gpus_available

def _to_tensor(x, dtype):
  return ops.convert_to_tensor(x, dtype=dtype)

@tf_export('keras.backend.is_sparse')
def is_sparse(tensor):
  return isinstance(tensor, sparse_tensor.SparseTensor)

@tf_export('keras.backend.to_dense')
def to_dense(tensor):
  if is_sparse(tensor):
    return sparse_ops.sparse_tensor_to_dense(tensor)
  else:
    return tensor

name_scope = ops.name_scope

@tf_export('keras.backend.variable')
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
    v._uses_learning_phase = False
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
  v._uses_learning_phase = False
  track_variable(v)
  return v

def track_tf_optimizer(tf_optimizer):
  if context.executing_eagerly():
    return
  graph = ops.get_default_graph()
  optimizers = _GRAPH_TF_OPTIMIZERS.setdefault(graph, weakref.WeakSet())
  optimizers.add(tf_optimizer)

def track_variable(v):
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else ops.get_default_graph()
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
  variables = _get_variables(ops.get_default_graph())
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

@tf_export('keras.backend.constant')
def constant(value, dtype=None, shape=None, name=None):
  if dtype is None:
    dtype = floatx()
  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)

def is_keras_tensor(x):
  if (not isinstance(x, (ops.Tensor,
                         variables_module.Variable,
                         sparse_tensor.SparseTensor)) and
      x.__class__.__name__ != 'DeferredTensor'):
    raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                     '`. Expected a symbolic tensor instance.')
  return hasattr(x, '_keras_history')

@tf_export('keras.backend.placeholder')
def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
  if dtype is None:
    dtype = floatx()
  if not shape:
    if ndim:
      shape = tuple([None for _ in range(ndim)])
  if sparse:
    x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
  else:
    x = array_ops.placeholder(dtype, shape=shape, name=name)
  x._uses_learning_phase = False
  return x

def is_placeholder(x):
  try:
    return x.op.type == 'Placeholder'
  except AttributeError:
    return False

@tf_export('keras.backend.shape')
def shape(x):
  return array_ops.shape(x)

@tf_export('keras.backend.int_shape')
def int_shape(x):
  try:
    shape = x.shape
    if not isinstance(shape, tuple):
      shape = tuple(shape.as_list())
    return shape
  except ValueError:
    return None

@tf_export('keras.backend.ndim')
def ndim(x):
  dims = x.shape._dims
  if dims is not None:
    return len(dims)
  return None

@tf_export('keras.backend.dtype')
def dtype(x):
  return x.dtype.base_dtype.name

@tf_export('keras.backend.eval')
def eval(x):
  return to_dense(x).eval(session=get_session())

@tf_export('keras.backend.zeros')
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

@tf_export('keras.backend.ones')
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

@tf_export('keras.backend.eye')
def eye(size, dtype=None, name=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  return variable(linalg_ops.eye(size, dtype=tf_dtype), dtype, name)

@tf_export('keras.backend.zeros_like')
def zeros_like(x, dtype=None, name=None):
  return array_ops.zeros_like(x, dtype=dtype, name=name)

@tf_export('keras.backend.ones_like')
def ones_like(x, dtype=None, name=None):
  return array_ops.ones_like(x, dtype=dtype, name=name)

def identity(x, name=None):
  return array_ops.identity(x, name=name)

@tf_export('keras.backend.random_uniform_variable')
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    seed = np.random.randint(10e8)
  value = init_ops.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)

@tf_export('keras.backend.random_normal_variable')
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

@tf_export('keras.backend.count_params')
def count_params(x):
  return np.prod(x.shape.as_list())

@tf_export('keras.backend.cast')
def cast(x, dtype):
  return math_ops.cast(x, dtype)

@tf_export('keras.backend.update')
def update(x, new_x):
  return state_ops.assign(x, new_x)

@tf_export('keras.backend.update_add')
def update_add(x, increment):
  return state_ops.assign_add(x, increment)

@tf_export('keras.backend.update_sub')
def update_sub(x, decrement):
  return state_ops.assign_sub(x, decrement)

@tf_export('keras.backend.moving_average_update')
def moving_average_update(x, value, momentum):
  from tensorflow.python.training import moving_averages  
  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=True)

@tf_export('keras.backend.dot')
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

@tf_export('keras.backend.batch_dot')
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

@tf_export('keras.backend.transpose')
def transpose(x):
  return array_ops.transpose(x)

@tf_export('keras.backend.gather')
def gather(reference, indices):
  return array_ops.gather(reference, indices)

@tf_export('keras.backend.max')
def max(x, axis=None, keepdims=False):
  return math_ops.reduce_max(x, axis, keepdims)

@tf_export('keras.backend.min')
def min(x, axis=None, keepdims=False):
  return math_ops.reduce_min(x, axis, keepdims)

@tf_export('keras.backend.sum')
def sum(x, axis=None, keepdims=False):
  return math_ops.reduce_sum(x, axis, keepdims)

@tf_export('keras.backend.prod')
def prod(x, axis=None, keepdims=False):
  return math_ops.reduce_prod(x, axis, keepdims)

def cumsum(x, axis=0):
  return math_ops.cumsum(x, axis=axis)

def cumprod(x, axis=0):
  return math_ops.cumprod(x, axis=axis)

@tf_export('keras.backend.var')
def var(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  m = math_ops.reduce_mean(x, axis, True)
  devs_squared = math_ops.square(x - m)
  return math_ops.reduce_mean(
      devs_squared, axis, keepdims)

@tf_export('keras.backend.std')
def std(x, axis=None, keepdims=False):
  return math_ops.sqrt(var(x, axis=axis, keepdims=keepdims))

@tf_export('keras.backend.mean')
def mean(x, axis=None, keepdims=False):
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, axis, keepdims)

@tf_export('keras.backend.any')
def any(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, axis, keepdims)

@tf_export('keras.backend.all')
def all(x, axis=None, keepdims=False):
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, axis, keepdims)

@tf_export('keras.backend.argmax')
def argmax(x, axis=-1):
  return math_ops.argmax(x, axis)

@tf_export('keras.backend.argmin')
def argmin(x, axis=-1):
  return math_ops.argmin(x, axis)

@tf_export('keras.backend.square')
def square(x):
  return math_ops.square(x)

@tf_export('keras.backend.abs')
def abs(x):
  return math_ops.abs(x)

@tf_export('keras.backend.sqrt')
def sqrt(x):
  zero = _to_tensor(0., x.dtype.base_dtype)
  inf = _to_tensor(np.inf, x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, inf)
  return math_ops.sqrt(x)

@tf_export('keras.backend.exp')
def exp(x):
  return math_ops.exp(x)

@tf_export('keras.backend.log')
def log(x):
  return math_ops.log(x)

def logsumexp(x, axis=None, keepdims=False):
  return math_ops.reduce_logsumexp(x, axis, keepdims)

@tf_export('keras.backend.round')
def round(x):
  return math_ops.round(x)

@tf_export('keras.backend.sign')
def sign(x):
  return math_ops.sign(x)

@tf_export('keras.backend.pow')
def pow(x, a):
  return math_ops.pow(x, a)

@tf_export('keras.backend.clip')
def clip(x, min_value, max_value):
  if max_value is not None and max_value < min_value:
    max_value = min_value
  if max_value is None:
    max_value = np.inf
  min_value = _to_tensor(min_value, x.dtype.base_dtype)
  max_value = _to_tensor(max_value, x.dtype.base_dtype)
  return clip_ops.clip_by_value(x, min_value, max_value)

@tf_export('keras.backend.equal')
def equal(x, y):
  return math_ops.equal(x, y)

@tf_export('keras.backend.not_equal')
def not_equal(x, y):
  return math_ops.not_equal(x, y)

@tf_export('keras.backend.greater')
def greater(x, y):
  return math_ops.greater(x, y)

@tf_export('keras.backend.greater_equal')
def greater_equal(x, y):
  return math_ops.greater_equal(x, y)

@tf_export('keras.backend.less')
def less(x, y):
  return math_ops.less(x, y)

@tf_export('keras.backend.less_equal')
def less_equal(x, y):
  return math_ops.less_equal(x, y)

@tf_export('keras.backend.maximum')
def maximum(x, y):
  return math_ops.maximum(x, y)

@tf_export('keras.backend.minimum')
def minimum(x, y):
  return math_ops.minimum(x, y)

@tf_export('keras.backend.sin')
def sin(x):
  return math_ops.sin(x)

@tf_export('keras.backend.cos')
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

@tf_export('keras.backend.normalize_batch_in_training')
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

@tf_export('keras.backend.batch_normalization')
def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

@tf_export('keras.backend.concatenate')
def concatenate(tensors, axis=-1):
  if axis < 0:
    rank = ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if py_all([is_sparse(x) for x in tensors]):
    return sparse_ops.sparse_concat(axis, tensors)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)

@tf_export('keras.backend.reshape')
def reshape(x, shape):
  return array_ops.reshape(x, shape)

@tf_export('keras.backend.permute_dimensions')
def permute_dimensions(x, pattern):
  return array_ops.transpose(x, perm=pattern)

@tf_export('keras.backend.resize_images')
def resize_images(x, height_factor, width_factor, data_format):
  if data_format == 'channels_first':
    original_shape = int_shape(x)
    new_shape = array_ops.shape(x)[2:]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = permute_dimensions(x, [0, 2, 3, 1])
    x = image_ops.resize_nearest_neighbor(x, new_shape)
    x = permute_dimensions(x, [0, 3, 1, 2])
    x.set_shape((None, None, original_shape[2] * height_factor
                 if original_shape[2] is not None else None,
                 original_shape[3] * width_factor
                 if original_shape[3] is not None else None))
    return x
  elif data_format == 'channels_last':
    original_shape = int_shape(x)
    new_shape = array_ops.shape(x)[1:3]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = image_ops.resize_nearest_neighbor(x, new_shape)
    x.set_shape((None, original_shape[1] * height_factor
                 if original_shape[1] is not None else None,
                 original_shape[2] * width_factor
                 if original_shape[2] is not None else None, None))
    return x
  else:
    raise ValueError('Invalid data_format: ' + str(data_format))

@tf_export('keras.backend.resize_volumes')
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

@tf_export('keras.backend.repeat_elements')
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

@tf_export('keras.backend.repeat')
def repeat(x, n):
  assert ndim(x) == 2
  x = array_ops.expand_dims(x, 1)
  pattern = array_ops.stack([1, n, 1])
  return array_ops.tile(x, pattern)

@tf_export('keras.backend.arange')
def arange(start, stop=None, step=1, dtype='int32'):
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result

def tile(x, n):
  if isinstance(n, int):
    n = [n]
  return array_ops.tile(x, n)

@tf_export('keras.backend.flatten')
def flatten(x):
  return array_ops.reshape(x, [-1])

@tf_export('keras.backend.batch_flatten')
def batch_flatten(x):
  x = array_ops.reshape(x, array_ops.stack([-1, prod(shape(x)[1:])]))
  return x

@tf_export('keras.backend.expand_dims')
def expand_dims(x, axis=-1):
  return array_ops.expand_dims(x, axis)

@tf_export('keras.backend.squeeze')
def squeeze(x, axis):
  return array_ops.squeeze(x, [axis])

@tf_export('keras.backend.temporal_padding')
def temporal_padding(x, padding=(1, 1)):
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return array_ops.pad(x, pattern)

@tf_export('keras.backend.spatial_2d_padding')
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

@tf_export('keras.backend.spatial_3d_padding')
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

@tf_export('keras.backend.stack')
def stack(x, axis=0):
  return array_ops.stack(x, axis=axis)

@tf_export('keras.backend.one_hot')
def one_hot(indices, num_classes):
  return array_ops.one_hot(indices, depth=num_classes, axis=-1)

@tf_export('keras.backend.reverse')
def reverse(x, axes):
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)

@tf_export('keras.backend.get_value')
def get_value(x):
  if context.executing_eagerly():
    return x.numpy()
  return x.eval(session=get_session())

@tf_export('keras.backend.batch_get_value')
def batch_get_value(tensors):
  if context.executing_eagerly():
    return [x.numpy() for x in tensors]
  if tensors:
    return get_session().run(tensors)
  else:
    return []

@tf_export('keras.backend.set_value')
def set_value(x, value):
  value = np.asarray(value, dtype=dtype(x))
  if context.executing_eagerly():
    x.assign(value)
  else:
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

@tf_export('keras.backend.batch_set_value')
def batch_set_value(tuples):
  if context.executing_eagerly():
    for x, value in tuples:
      x.assign(np.asarray(value, dtype=dtype(x)))
  else:
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

@tf_export('keras.backend.print_tensor')
def print_tensor(x, message=''):
  return logging_ops.Print(x, [x], message)

class Function(object):

  def __init__(self, inputs, outputs, updates=None, name=None,
               **session_kwargs):
    updates = updates or []
    if not isinstance(inputs, (list, tuple)):
      raise TypeError('`inputs` to a TensorFlow backend function '
                      'should be a list or tuple.')
    if not isinstance(outputs, (list, tuple)):
      raise TypeError('`outputs` of a TensorFlow backend function '
                      'should be a list or tuple.')
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a TensorFlow backend function '
                      'should be a list or tuple.')
    self.inputs = list(inputs)
    self.outputs = list(outputs)
    with ops.control_dependencies(self.outputs):
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
    self.fetch_callbacks = dict()

    if session_kwargs:
      raise ValueError('Some keys in session_kwargs are not supported at this '
                       'time: %s', session_kwargs.keys())

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

  def __call__(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise TypeError('`inputs` should be a list or tuple.')

    session = get_session()
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
    return fetched[:len(self.outputs)]

@tf_export('keras.backend.function')
def function(inputs, outputs, updates=None, **kwargs):
  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in tf_inspect.getfullargspec(Function.__init__)[0]):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return Function(inputs, outputs, updates=updates, **kwargs)

@tf_export('keras.backend.gradients')
def gradients(loss, variables):
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)

@tf_export('keras.backend.stop_gradient')
def stop_gradient(variables):
  if isinstance(variables, (list, tuple)):
    return map(array_ops.stop_gradient, variables)
  return array_ops.stop_gradient(variables)

@tf_export('keras.backend.rnn')
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None):
  ndim = len(inputs.shape)
  if ndim < 3:
    raise ValueError('Input should be at least 3D.')
  inputs_shape = inputs.shape
  axes = [1, 0] + list(range(2, ndim))
  inputs = array_ops.transpose(inputs, (axes))

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == ndim - 1:
      mask = expand_dims(mask)
    mask = array_ops.transpose(mask, axes)

  if constants is None:
    constants = []

  global uses_learning_phase  
  uses_learning_phase = False

  if unroll:
    if not inputs.shape[0]:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = initial_states
    successive_states = []
    successive_outputs = []

    input_list = array_ops.unstack(inputs)
    if go_backwards:
      input_list.reverse()

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for inp, mask_t in zip(input_list, mask_list):
        output, new_states = step_function(inp, states + constants)
        if getattr(output, '_uses_learning_phase', False):
          uses_learning_phase = True

        tiled_mask_t = array_ops.tile(mask_t,
                                      array_ops.stack(
                                          [1, array_ops.shape(output)[1]]))

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where(tiled_mask_t, output, prev_output)

        return_states = []
        for state, new_state in zip(states, new_states):
          tiled_mask_t = array_ops.tile(mask_t,
                                        array_ops.stack(
                                            [1,
                                             array_ops.shape(new_state)[1]]))
          return_states.append(array_ops.where(tiled_mask_t, new_state, state))
        states = return_states
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)
    else:
      for inp in input_list:
        output, states = step_function(inp, states + constants)
        if getattr(output, '_uses_learning_phase', False):
          uses_learning_phase = True
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:
    if go_backwards:
      inputs = reverse(inputs, 0)

    states = tuple(initial_states)

    time_steps = array_ops.shape(inputs)[0]
    outputs, _ = step_function(inputs[0], initial_states + constants)
    output_ta = tensor_array_ops.TensorArray(
        dtype=outputs.dtype, size=time_steps, tensor_array_name='output_ta')
    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype, size=time_steps, tensor_array_name='input_ta')
    input_ta = input_ta.unstack(inputs)
    time = constant_op.constant(0, dtype='int32', name='time')

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
          size=time_steps,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def _step(time, output_ta_t, *states):
        current_input = input_ta.read(time)
        mask_t = mask_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        if getattr(output, '_uses_learning_phase', False):
          global uses_learning_phase  
          uses_learning_phase = True
        for state, new_state in zip(states, new_states):
          new_state.set_shape(state.shape)
        tiled_mask_t = array_ops.tile(mask_t,
                                      array_ops.stack(
                                          [1, array_ops.shape(output)[1]]))
        output = array_ops.where(tiled_mask_t, output, states[0])

        masked_states = []
        for i in range(len(states)):
          states_dim = array_ops.shape(new_states[i])[1]
          stacked_states_dim = array_ops.stack([1, states_dim])
          tiled_mask = array_ops.tile(mask_t, stacked_states_dim)
          masked_state = array_ops.where(tiled_mask, new_states[i], states[i])
          masked_states.append(masked_state)
        new_states = masked_states

        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)
    else:

      def _step(time, output_ta_t, *states):
        current_input = input_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        if getattr(output, '_uses_learning_phase', False):
          global uses_learning_phase  
          uses_learning_phase = True
        for state, new_state in zip(states, new_states):
          new_state.set_shape(state.shape)
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_step,
        loop_vars=(time, output_ta) + states,
        maximum_iterations=input_length,
        parallel_iterations=32,
        swap_memory=True)
    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    new_states = final_outputs[2:]

    outputs = output_ta.stack()
    last_output = output_ta.read(last_time - 1)

  axes = [1, 0] + list(range(2, len(outputs.shape)))
  outputs = array_ops.transpose(outputs, axes)

  outputs_shape = outputs.shape.as_list()
  outputs_shape[0] = inputs_shape[0]
  outputs_shape[1] = inputs_shape[1]
  outputs.set_shape(outputs_shape)

  if not context.executing_eagerly():
    last_output._uses_learning_phase = uses_learning_phase
  return last_output, outputs, new_states

@tf_export('keras.backend.switch')
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

@tf_export('keras.backend.in_train_phase')
def in_train_phase(x, alt, training=None):
  if training is None:
    training = learning_phase()
    uses_learning_phase = True
  else:
    uses_learning_phase = False

  if training is 1 or training is True:
    if callable(x):
      return x()
    else:
      return x

  elif training is 0 or training is False:
    if callable(alt):
      return alt()
    else:
      return alt

  x = switch(training, x, alt)
  if uses_learning_phase:
    x._uses_learning_phase = True
  return x

@tf_export('keras.backend.in_test_phase')
def in_test_phase(x, alt, training=None):
  return in_train_phase(alt, x, training=training)

@tf_export('keras.backend.relu')
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
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    zero = _to_tensor(0., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x

@tf_export('keras.backend.elu')
def elu(x, alpha=1.):
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where(x > 0, res, alpha * res)

@tf_export('keras.backend.softmax')
def softmax(x, axis=-1):
  return nn.softmax(x, axis=axis)

@tf_export('keras.backend.softplus')
def softplus(x):
  return nn.softplus(x)

@tf_export('keras.backend.softsign')
def softsign(x):
  return nn.softsign(x)

@tf_export('keras.backend.categorical_crossentropy')
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  rank = len(output.shape)
  axis = axis % rank
  if not from_logits:
    output = output / math_ops.reduce_sum(output, axis, True)
    epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return -math_ops.reduce_sum(target * math_ops.log(output), axis)
  else:
    return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)

@tf_export('keras.backend.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  rank = len(output.shape)
  axis = axis % rank
  if axis != rank - 1:
    permutation = list(range(axis)) + list(range(axis + 1, rank)) + [axis]
    output = array_ops.transpose(output, perm=permutation)

  if not from_logits:
    epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output)

  output_shape = output.shape
  targets = cast(flatten(target), 'int64')
  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
  res = nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
  if len(output_shape) >= 3:
    return array_ops.reshape(res, array_ops.shape(output)[:-1])
  else:
    return res

@tf_export('keras.backend.binary_crossentropy')
def binary_crossentropy(target, output, from_logits=False):
  if not from_logits:
    epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output / (1 - output))
  return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

@tf_export('keras.backend.sigmoid')
def sigmoid(x):
  return nn.sigmoid(x)

@tf_export('keras.backend.hard_sigmoid')
def hard_sigmoid(x):
  x = (0.2 * x) + 0.5
  zero = _to_tensor(0., x.dtype.base_dtype)
  one = _to_tensor(1., x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, one)
  return x

@tf_export('keras.backend.tanh')
def tanh(x):
  return nn.tanh(x)

@tf_export('keras.backend.dropout')
def dropout(x, level, noise_shape=None, seed=None):
  retain_prob = 1. - level
  if seed is None:
    seed = np.random.randint(10e6)
  return nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)

@tf_export('keras.backend.l2_normalize')
def l2_normalize(x, axis=None):
  return nn.l2_normalize(x, axis=axis)

@tf_export('keras.backend.in_top_k')
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

def _preprocess_conv2d_input(x, data_format):
  tf_data_format = 'NHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support():
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

@tf_export('keras.backend.conv1d')
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
      dilation_rate=(dilation_rate,),
      strides=(strides,),
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  
  return x

@tf_export('keras.backend.conv2d')
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

@tf_export('keras.backend.conv2d_transpose')
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None):
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(output_shape)

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[1])
  if output_shape[0] is None:
    output_shape = (array_ops.shape(x)[0],) + tuple(output_shape[1:])
    output_shape = array_ops.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.conv2d_transpose(
      x,
      kernel,
      output_shape,
      strides,
      padding=padding,
      data_format=tf_data_format)
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

@tf_export('keras.backend.separable_conv2d')
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

@tf_export('keras.backend.conv3d')
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

@tf_export('keras.backend.pool2d')
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

@tf_export('keras.backend.pool3d')
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

def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  output_shape = (kernel.shape[0],)
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)

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

@tf_export('keras.backend.bias_add')
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

@tf_export('keras.backend.random_normal')
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_normal(
      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

@tf_export('keras.backend.random_uniform')
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_uniform(
      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)

@tf_export('keras.backend.random_binomial')
def random_binomial(shape, p=0.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return array_ops.where(
      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))

@tf_export('keras.backend.truncated_normal')
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.truncated_normal(
      shape, mean, stddev, dtype=dtype, seed=seed)

@tf_export('keras.backend.ctc_label_dense_to_sparse')
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
      math_ops.to_int64(indices), vals_sparse, math_ops.to_int64(label_shape))

@tf_export('keras.backend.ctc_batch_cost')
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
  label_length = math_ops.to_int32(array_ops.squeeze(label_length, axis=-1))
  input_length = math_ops.to_int32(array_ops.squeeze(input_length, axis=-1))
  sparse_labels = math_ops.to_int32(
      ctc_label_dense_to_sparse(y_true, label_length))

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)

@tf_export('keras.backend.ctc_decode')
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
  input_length = math_ops.to_int32(input_length)

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

@tf_export('keras.backend.map_fn')
def map_fn(fn, elems, name=None, dtype=None):
  return functional_ops.map_fn(fn, elems, name=name, dtype=dtype)

@tf_export('keras.backend.foldl')
def foldl(fn, elems, initializer=None, name=None):
  return functional_ops.foldl(fn, elems, initializer=initializer, name=name)

@tf_export('keras.backend.foldr')
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
EOF
