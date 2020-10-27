

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import sys
import types
import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
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
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

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
        obj.update_state = metrics_utils.weakmethod(obj.update_state)
      obj.update_state = metrics_utils.weakmethod(
          types.MethodType(
              metrics_utils.update_state_wrapper(obj.update_state), obj))
      result = metrics_utils.weakmethod(obj.result)
      obj.result = metrics_utils.weakmethod(
          types.MethodType(metrics_utils.result_wrapper(result), obj))
    else:
      obj.update_state = types.MethodType(
          metrics_utils.update_state_wrapper(obj.update_state), obj)
      obj.result = types.MethodType(
          metrics_utils.result_wrapper(obj.result), obj)

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

@keras_export('keras.metrics.Mean')
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

@keras_export('keras.metrics.Accuracy')
class Accuracy(MeanMetricWrapper):

  def __init__(self, name='accuracy', dtype=None):
    super(Accuracy, self).__init__(accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(Accuracy, cls).from_config(config)

@keras_export('keras.metrics.BinaryAccuracy')
class BinaryAccuracy(MeanMetricWrapper):

  def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
    super(BinaryAccuracy, self).__init__(
        binary_accuracy, name, dtype=dtype, threshold=threshold)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(BinaryAccuracy, cls).from_config(config)

@keras_export('keras.metrics.CategoricalAccuracy')
class CategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='categorical_accuracy', dtype=None):
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CategoricalAccuracy, cls).from_config(config)

@keras_export('keras.metrics.SparseCategoricalAccuracy')
class SparseCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SparseCategoricalAccuracy, cls).from_config(config)

class TopKCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
    super(TopKCategoricalAccuracy, self).__init__(
        top_k_categorical_accuracy, name, dtype=dtype, k=k)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(TopKCategoricalAccuracy, cls).from_config(config)

class SparseTopKCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, k=5, name='sparse_top_k_categorical_accuracy', dtype=None):
    super(SparseTopKCategoricalAccuracy, self).__init__(
        sparse_top_k_categorical_accuracy, name, dtype=dtype, k=k)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SparseTopKCategoricalAccuracy, cls).from_config(config)

class _ConfusionMatrixConditionCount(Metric):

  def __init__(self,
               confusion_matrix_cond,
               thresholds=None,
               name=None,
               dtype=None):
    super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
    self._confusion_matrix_cond = confusion_matrix_cond
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.accumulator = self.add_weight(
        'accumulator',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return metrics_utils.update_confusion_matrix_variables(
        {self._confusion_matrix_cond: self.accumulator}, y_true, y_pred,
        self.thresholds, sample_weight)

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

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(_ConfusionMatrixConditionCount, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.FalsePositives')
class FalsePositives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(FalsePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@keras_export('keras.metrics.FalseNegatives')
class FalseNegatives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(FalseNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@keras_export('keras.metrics.TrueNegatives')
class TrueNegatives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(TrueNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@keras_export('keras.metrics.TruePositives')
class TruePositives(_ConfusionMatrixConditionCount):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(TruePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)

@keras_export('keras.metrics.Precision')
class Precision(Metric):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(Precision, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
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
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fp)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(Precision, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.Recall')
class Recall(Metric):

  def __init__(self, thresholds=None, name=None, dtype=None):
    super(Recall, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
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
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fn)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(Recall, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

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
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.tn,
        metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp,
        metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn,
    }, y_true, y_pred, self.thresholds, sample_weight)

  def reset_states(self):
    num_thresholds = len(self.thresholds)
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

@keras_export('keras.metrics.SensitivityAtSpecificity')
class SensitivityAtSpecificity(SensitivitySpecificityBase):

  def __init__(self, specificity, num_thresholds=200, name=None, dtype=None):
    if specificity < 0 or specificity > 1:
      raise ValueError('`specificity` must be in the range [0, 1].')
    self.specificity = specificity
    self.num_thresholds = num_thresholds
    super(SensitivityAtSpecificity, self).__init__(
        specificity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    specificities = math_ops.div_no_nan(self.tn, self.tn + self.fp)

    min_index = math_ops.argmin(
        math_ops.abs(specificities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(self.tp[min_index],
                               self.tp[min_index] + self.fn[min_index])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'specificity': self.specificity
    }
    base_config = super(SensitivityAtSpecificity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.SpecificityAtSensitivity')
class SpecificityAtSensitivity(SensitivitySpecificityBase):

  def __init__(self, sensitivity, num_thresholds=200, name=None, dtype=None):
    if sensitivity < 0 or sensitivity > 1:
      raise ValueError('`sensitivity` must be in the range [0, 1].')
    self.sensitivity = sensitivity
    self.num_thresholds = num_thresholds
    super(SpecificityAtSensitivity, self).__init__(
        sensitivity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    sensitivities = math_ops.div_no_nan(self.tp, self.tp + self.fn)

    min_index = math_ops.argmin(
        math_ops.abs(sensitivities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(self.tn[min_index],
                               self.tn[min_index] + self.fp[min_index])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'sensitivity': self.sensitivity
    }
    base_config = super(SpecificityAtSensitivity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.CosineProximity')
class CosineProximity(MeanMetricWrapper):

  def __init__(self, name='cosine_proximity', dtype=None):
    super(CosineProximity, self).__init__(cosine, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CosineProximity, cls).from_config(config)

@keras_export('keras.metrics.MeanAbsoluteError')
class MeanAbsoluteError(MeanMetricWrapper):

  def __init__(self, name='mean_absolute_error', dtype=None):
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanAbsoluteError, cls).from_config(config)

@keras_export('keras.metrics.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(MeanMetricWrapper):

  def __init__(self, name='mean_absolute_percentage_error', dtype=None):
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanAbsolutePercentageError, cls).from_config(config)

@keras_export('keras.metrics.MeanSquaredError')
class MeanSquaredError(MeanMetricWrapper):

  def __init__(self, name='mean_squared_error', dtype=None):
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanSquaredError, cls).from_config(config)

@keras_export('keras.metrics.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(MeanMetricWrapper):

  def __init__(self, name='mean_squared_logarithmic_error', dtype=None):
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanSquaredLogarithmicError, cls).from_config(config)

@keras_export('keras.metrics.Hinge')
class Hinge(MeanMetricWrapper):

  def __init__(self, name='hinge', dtype=None):
    super(Hinge, self).__init__(hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(Hinge, cls).from_config(config)

@keras_export('keras.metrics.SquaredHinge')
class SquaredHinge(MeanMetricWrapper):

  def __init__(self, name='squared_hinge', dtype=None):
    super(SquaredHinge, self).__init__(squared_hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SquaredHinge, cls).from_config(config)

@keras_export('keras.metrics.CategoricalHinge')
class CategoricalHinge(MeanMetricWrapper):

  def __init__(self, name='categorical_hinge', dtype=None):
    super(CategoricalHinge, self).__init__(categorical_hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CategoricalHinge, cls).from_config(config)

class RootMeanSquaredError(Mean):

  def __init__(self, name='root_mean_squared_error', dtype=None):
    super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)
    error_sq = math_ops.square(y_pred - y_true)
    return super(RootMeanSquaredError, self).update_state(
        error_sq, sample_weight=sample_weight)

  def result(self):
    return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))

def accuracy(y_true, y_pred):
  y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
  if y_true.dtype != y_pred.dtype:
    y_pred = math_ops.cast(y_pred, y_true.dtype)
  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

@keras_export('keras.metrics.binary_accuracy')
def binary_accuracy(y_true, y_pred, threshold=0.5):
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

@keras_export('keras.metrics.categorical_accuracy')
def categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      K.floatx())

@keras_export('keras.metrics.sparse_categorical_accuracy')
def sparse_categorical_accuracy(y_true, y_pred):
  y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])
  y_pred = math_ops.argmax(y_pred, axis=-1)

  if K.dtype(y_pred) != K.dtype(y_true):
    y_pred = math_ops.cast(y_pred, K.dtype(y_true))

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

@keras_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), axis=-1)

@keras_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])

  return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), axis=-1)

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity

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

@keras_export('keras.metrics.serialize')
def serialize(metric):
  return serialize_keras_object(metric)

@keras_export('keras.metrics.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')

@keras_export('keras.metrics.get')
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
