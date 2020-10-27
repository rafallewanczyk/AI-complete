

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import sys
import types
import weakref
from enum import Enum
import numpy as np
import six

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import cosine_proximity
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

def clone_metric(metric):
  if isinstance(metric, Metric):
    return metric.__class__.from_config(metric.get_config())
  return metric

def clone_metrics(metrics):
  if metrics is None:
    return None
  if isinstance(metrics, dict):
    return {key: clone_metric(value) for key, value in metrics.items()}
  return [clone_metric(metric) for metric in metrics]

def update_state_wrapper(update_state_fn):

  def decorated(metric_obj, *args, **kwargs):

    update_op = update_state_fn(*args, **kwargs)
    if update_op is not None:  
      metric_obj.add_update(update_op, inputs=True)
    return update_op

  return tf_decorator.make_decorator(update_state_fn, decorated)

def result_wrapper(result_fn):

  def decorated(_, *args):
    replica_context = distribution_strategy_context.get_replica_context()
    if replica_context is None:  
      result_t = result_fn(*args)
    else:

      def merge_fn_wrapper(distribution, merge_fn, *args):
        return distribution.unwrap(merge_fn)[0](*args)

      result_t = replica_context.merge_call(
          merge_fn_wrapper, args=(result_fn,) + args)
    return result_t

  return tf_decorator.make_decorator(result_fn, decorated)

def weakmethod(method):

  cls = method.im_class
  func = method.im_func
  instance_ref = weakref.ref(method.im_self)

  @functools.wraps(method)
  def inner(*args, **kwargs):
    return func.__get__(instance_ref(), cls)(*args, **kwargs)

  del method
  return inner

class _ConfusionMatrix(Enum):
  TRUE_POSITIVES = 'tp'
  FALSE_POSITIVES = 'fp'
  TRUE_NEGATIVES = 'tn'
  FALSE_NEGATIVES = 'fn'

def _assert_thresholds_range(thresholds):
  invalid_thresholds = [t for t in thresholds if t is None or t < 0 or t > 1]
  if invalid_thresholds:
    raise ValueError('Threshold values must be in [0, 1]. Invalid values: {}'
                     .format(invalid_thresholds))

def _parse_init_thresholds(thresholds, default_threshold=0.5):
  thresholds = to_list(default_threshold if thresholds is None else thresholds)
  _assert_thresholds_range(thresholds)
  return thresholds

def _update_confusion_matrix_variables(variables_to_update,
                                       y_true,
                                       y_pred,
                                       thresholds,
                                       sample_weight=None):
  if variables_to_update is None:
    return
  y_true = ops.convert_to_tensor(y_true)
  y_pred = ops.convert_to_tensor(y_pred)
  y_pred.shape.assert_is_compatible_with(y_true.shape)

  if not any(
      key for key in variables_to_update if key in list(_ConfusionMatrix)):
    raise ValueError(
        'Please provide at least one valid confusion matrix '
        'variable to update. Valid variable key options are: "{}". '
        'Received: "{}"'.format(
            list(_ConfusionMatrix), variables_to_update.keys()))

  invalid_keys = [
      key for key in variables_to_update if key not in list(_ConfusionMatrix)
  ]
  if invalid_keys:
    raise ValueError(
        'Invalid keys: {}. Valid variable key options are: "{}"'.format(
            invalid_keys, list(_ConfusionMatrix)))

  with ops.control_dependencies([
      check_ops.assert_greater_equal(
          y_pred,
          math_ops.cast(0.0, dtype=y_pred.dtype),
          message='predictions must be >= 0'),
      check_ops.assert_less_equal(
          y_pred,
          math_ops.cast(1.0, dtype=y_pred.dtype),
          message='predictions must be <= 1')
  ]):
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        math_ops.cast(y_pred, dtype=dtypes.float32),
        math_ops.cast(y_true, dtype=dtypes.bool), sample_weight)

  thresholds = to_list(thresholds)
  num_thresholds = len(thresholds)
  num_predictions = array_ops.size(y_pred)

  predictions_2d = array_ops.reshape(y_pred, [1, -1])
  labels_2d = array_ops.reshape(
      math_ops.cast(y_true, dtype=dtypes.bool), [1, -1])

  thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(thresholds), 1),
      array_ops.stack([1, num_predictions]))

  preds_tiled = array_ops.tile(predictions_2d, [num_thresholds, 1])

  pred_is_pos = math_ops.greater(preds_tiled, thresh_tiled)

  label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])

  if sample_weight is not None:
    weights = weights_broadcast_ops.broadcast_weights(
        math_ops.cast(sample_weight, dtype=dtypes.float32), y_pred)
    weights_tiled = array_ops.tile(
        array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
  else:
    weights_tiled = None

  update_ops = []

  def weighted_assign_add(label, pred, weights, var):
    label_and_pred = math_ops.cast(
        math_ops.logical_and(label, pred), dtype=dtypes.float32)
    if weights is not None:
      label_and_pred *= weights
    return state_ops.assign_add(var, math_ops.reduce_sum(label_and_pred, 1))

  loop_vars = {
      _ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
  }
  update_tn = _ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
  update_fp = _ConfusionMatrix.FALSE_POSITIVES in variables_to_update
  update_fn = _ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

  if update_fn or update_tn:
    pred_is_neg = math_ops.logical_not(pred_is_pos)
    loop_vars[_ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

  if update_fp or update_tn:
    label_is_neg = math_ops.logical_not(label_is_pos)
    loop_vars[_ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
    if update_tn:
      loop_vars[_ConfusionMatrix.TRUE_NEGATIVES] = (label_is_neg, pred_is_neg)

  for matrix_cond, (label, pred) in loop_vars.items():
    if matrix_cond in variables_to_update:
      update_ops.append(
          weighted_assign_add(label, pred, weights_tiled,
                              variables_to_update[matrix_cond]))
  return control_flow_ops.group(update_ops)

@six.add_metaclass(abc.ABCMeta)
class Metric(Layer):

  def __init__(self, name=None, dtype=None):
    super(Metric, self).__init__(name=name, dtype=dtype)
    self.stateful = True  
    self.built = True
    self._dtype = K.floatx() if dtype is None else dtypes.as_dtype(dtype).name

  def __new__(cls, *args, **kwargs):
    obj = super(Metric, cls).__new__(cls)

    if sys.version_info < (3,):
      if context.executing_eagerly():
        obj.update_state = weakmethod(obj.update_state)
      obj.update_state = weakmethod(
          types.MethodType(update_state_wrapper(obj.update_state), obj))
      result = weakmethod(obj.result)
      obj.result = weakmethod(types.MethodType(result_wrapper(result), obj))
    else:
      obj.update_state = types.MethodType(
          update_state_wrapper(obj.update_state), obj)
      obj.result = types.MethodType(result_wrapper(obj.result), obj)

    return obj

  def __call__(self, *args, **kwargs):
    update_op = self.update_state(*args, **kwargs)
    with ops.control_dependencies([update_op]):
      result_t = self.result()

      if not context.executing_eagerly():
        result_t._metric_obj = self  
      return result_t

  def reset_states(self):
    for v in self.variables:
      K.set_value(v, 0)

  @abc.abstractmethod
  def update_state(self, *args, **kwargs):
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def result(self):
    NotImplementedError('Must be implemented in subclasses.')

  @classmethod
  def from_config(cls, config):
    if 'trainable' in config:
      config.pop('trainable')
    return cls(**config)

  @doc_controls.for_subclass_implementers
  def add_weight(self,
                 name,
                 shape=(),
                 aggregation=tf_variables.VariableAggregation.SUM,
                 synchronization=tf_variables.VariableSynchronization.ON_READ,
                 initializer=None):
    return super(Metric, self).add_weight(
        name=name,
        shape=shape,
        dtype=self._dtype,
        trainable=False,
        initializer=initializer,
        collections=[],
        synchronization=synchronization,
        aggregation=aggregation)

@tf_export('keras.metrics.Mean')
class Mean(Metric):

  def __init__(self, name='mean', dtype=None):
    super(Mean, self).__init__(name=name, dtype=dtype)
    self.total = self.add_weight(
        'total', initializer=init_ops.zeros_initializer)
    self.count = self.add_weight(
        'count', initializer=init_ops.zeros_initializer)

  def update_state(self, values, sample_weight=None):
    values = math_ops.cast(values, self._dtype)
    if sample_weight is None:
      num_values = math_ops.cast(array_ops.size(values), self._dtype)
    else:
      sample_weight = math_ops.cast(sample_weight, self._dtype)

      values, _, sample_weight = squeeze_or_expand_dimensions(
          values, None, sample_weight)
      try:
        sample_weight = weights_broadcast_ops.broadcast_weights(
            sample_weight, values)
      except ValueError:
        ndim = K.ndim(values)
        weight_ndim = K.ndim(sample_weight)
        values = math_ops.reduce_mean(
            values, axis=list(range(weight_ndim, ndim)))

      num_values = math_ops.reduce_sum(sample_weight)
      values = math_ops.multiply(values, sample_weight)
    values = math_ops.reduce_sum(values)

    update_total_op = state_ops.assign_add(self.total, values)
    with ops.control_dependencies([update_total_op]):
      return state_ops.assign_add(self.count, num_values)

  def result(self):
    return math_ops.div_no_nan(self.total, self.count)

class MeanMetricWrapper(Mean):

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)

    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {'fn': self._fn}
    config.update(self._fn_kwargs)
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf_export('keras.metrics.Accuracy')
class Accuracy(MeanMetricWrapper):

  def __init__(self, name='accuracy', dtype=None):
    super(Accuracy, self).__init__(accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(Accuracy, cls).from_config(config)

@tf_export('keras.metrics.BinaryAccuracy')
class BinaryAccuracy(MeanMetricWrapper):

  def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
    super(BinaryAccuracy, self).__init__(
        binary_accuracy, name, dtype=dtype, threshold=threshold)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(BinaryAccuracy, cls).from_config(config)

@tf_export('keras.metrics.CategoricalAccuracy')
class CategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='categorical_accuracy', dtype=None):
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CategoricalAccuracy, cls).from_config(config)

@tf_export('keras.metrics.SparseCategoricalAccuracy')
class SparseCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SparseCategoricalAccuracy, cls).from_config(config)

class _ConfusionMatrixConditionCount(Metric):

  def __init__(self,
               confusion_matrix_cond,
               thresholds=None,
               name=None,
               dtype=None):
    super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
    self._confusion_matrix_cond = confusion_matrix_cond
    self.thresholds = _parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.accumulator = self.add_weight(
        'accumulator',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _update_confusion_matrix_variables({
        self._confusion_matrix_cond: self.accumulator
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    if len(self.thresholds) == 1:
      result = self.accumulator[0]
    else:
      result = self.accumulator
    return ops.convert_to_tensor(result)

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

@tf_export('keras.metrics.FalsePositives')
class FalsePositives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(FalsePositives, self).__init__(
        confusion_matrix_cond=_ConfusionMatrix.FALSE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@tf_export('keras.metrics.FalseNegatives')
class FalseNegatives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(FalseNegatives, self).__init__(
        confusion_matrix_cond=_ConfusionMatrix.FALSE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@tf_export('keras.metrics.TrueNegatives')
class TrueNegatives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(TrueNegatives, self).__init__(
        confusion_matrix_cond=_ConfusionMatrix.TRUE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@tf_export('keras.metrics.TruePositives')
class TruePositives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(TruePositives, self).__init__(
        confusion_matrix_cond=_ConfusionMatrix.TRUE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@tf_export('keras.metrics.Precision')
class Precision(Metric):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(Precision, self).__init__(name=name, dtype=dtype)
    self.thresholds = _parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.tp = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.fp = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _update_confusion_matrix_variables({
        _ConfusionMatrix.TRUE_POSITIVES: self.tp,
        _ConfusionMatrix.FALSE_POSITIVES: self.fp
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fp)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

@tf_export('keras.metrics.Recall')
class Recall(Metric):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(Recall, self).__init__(name=name, dtype=dtype)
    self.thresholds = _parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.tp = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.fn = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _update_confusion_matrix_variables({
        _ConfusionMatrix.TRUE_POSITIVES: self.tp,
        _ConfusionMatrix.FALSE_NEGATIVES: self.fn
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fn)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

@six.add_metaclass(abc.ABCMeta)
class SensitivitySpecificityBase(Metric):

  def __init__(self, value, num_thresholds=200, name=None, dtype=None):
    super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
    if num_thresholds <= 0:
      raise ValueError('`num_thresholds` must be > 0.')
    self.value = value
    self.tp = self.add_weight(
        'true_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.tn = self.add_weight(
        'true_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.fp = self.add_weight(
        'false_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.fn = self.add_weight(
        'false_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)

    if num_thresholds == 1:
      self.thresholds = [0.5]
    else:
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]
      self.thresholds = [0.0] + thresholds + [1.0]

  def update_state(self, y_true, y_pred, sample_weight=None):
    return _update_confusion_matrix_variables({
        _ConfusionMatrix.TRUE_POSITIVES: self.tp,
        _ConfusionMatrix.TRUE_NEGATIVES: self.tn,
        _ConfusionMatrix.FALSE_POSITIVES: self.fp,
        _ConfusionMatrix.FALSE_NEGATIVES: self.fn,
    }, y_true, y_pred, self.thresholds, sample_weight)

  def reset_states(self):
    num_thresholds = len(self.thresholds)
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

@tf_export('keras.metrics.SensitivityAtSpecificity')
class SensitivityAtSpecificity(SensitivitySpecificityBase):

  def __init__(self, specificity, num_thresholds=200, name=None, dtype=None):
    if specificity < 0 or specificity > 1:
      raise ValueError('`specificity` must be in the range [0, 1].')
    super(SensitivityAtSpecificity, self).__init__(
        specificity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    specificities = math_ops.div_no_nan(self.tn, self.tn + self.fp)

    min_index = math_ops.argmin(
        math_ops.abs(specificities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(self.tp[min_index],
                               self.tp[min_index] + self.fn[min_index])

@tf_export('keras.metrics.SpecificityAtSensitivity')
class SpecificityAtSensitivity(SensitivitySpecificityBase):

  def __init__(self, sensitivity, num_thresholds=200, name=None, dtype=None):
    if sensitivity < 0 or sensitivity > 1:
      raise ValueError('`sensitivity` must be in the range [0, 1].')
    super(SpecificityAtSensitivity, self).__init__(
        sensitivity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    sensitivities = math_ops.div_no_nan(self.tp, self.tp + self.fn)

    min_index = math_ops.argmin(
        math_ops.abs(sensitivities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(self.tn[min_index],
                               self.tn[min_index] + self.fp[min_index])

class CosineProximity(MeanMetricWrapper):

  def __init__(self, name='cosine_proximity', dtype=None):
    super(CosineProximity, self).__init__(cosine, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CosineProximity, cls).from_config(config)

def accuracy(y_true, y_pred):
  y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
  if y_true.dtype != y_pred.dtype:
    y_pred = math_ops.cast(y_pred, y_true.dtype)
  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

@tf_export('keras.metrics.binary_accuracy')
def binary_accuracy(y_true, y_pred, threshold=0.5):
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

@tf_export('keras.metrics.categorical_accuracy')
def categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      K.floatx())

@tf_export('keras.metrics.sparse_categorical_accuracy')
def sparse_categorical_accuracy(y_true, y_pred):
  if (len(K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])
  y_pred = math_ops.argmax(y_pred, axis=-1)

  if K.dtype(y_pred) != K.dtype(y_true):
    y_pred = math_ops.cast(y_pred, K.dtype(y_true))

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

@tf_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), axis=-1)

@tf_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  if (len(K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])

  return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), axis=-1)

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity

@tf_export('keras.metrics.serialize')
def serialize(metric):
  return serialize_keras_object(metric)

@tf_export('keras.metrics.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')

@tf_export('keras.metrics.get')
def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'metric function identifier: %s' % identifier)
EOF
