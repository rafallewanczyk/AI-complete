

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import types
import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import cosine_similarity
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
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

@keras_export('keras.metrics.Metric')
@six.add_metaclass(abc.ABCMeta)
class Metric(Layer):

  def __init__(self, name=None, dtype=None, **kwargs):
    super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
    self.stateful = True  
    self.built = True
    self._dtype = K.floatx() if dtype is None else dtypes.as_dtype(dtype).name

  def __new__(cls, *args, **kwargs):
    obj = super(Metric, cls).__new__(cls)

    if cls.__module__ == Metric.__module__:
      update_state_fn = obj.update_state
    else:
      update_state_fn = def_function.function(obj.update_state)

    obj.update_state = types.MethodType(
        metrics_utils.update_state_wrapper(update_state_fn), obj)
    obj.result = types.MethodType(metrics_utils.result_wrapper(obj.result), obj)
    return obj

  def __call__(self, *args, **kwargs):
    update_op = self.update_state(*args, **kwargs)  
    with ops.control_dependencies([update_op]):
      result_t = self.result()  

      result_t._metric_obj = self  
      return result_t

  @property
  def dtype(self):
    return self._dtype

  def get_config(self):
    return {'name': self.name, 'dtype': self.dtype}

  def reset_states(self):
    K.batch_set_value([(v, 0) for v in self.variables])

  @abc.abstractmethod
  def update_state(self, *args, **kwargs):
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def result(self):
    NotImplementedError('Must be implemented in subclasses.')

  @doc_controls.for_subclass_implementers
  def add_weight(self,
                 name,
                 shape=(),
                 aggregation=tf_variables.VariableAggregation.SUM,
                 synchronization=tf_variables.VariableSynchronization.ON_READ,
                 initializer=None,
                 dtype=None):
    return super(Metric, self).add_weight(
        name=name,
        shape=shape,
        dtype=self._dtype if dtype is None else dtype,
        trainable=False,
        initializer=initializer,
        collections=[],
        synchronization=synchronization,
        aggregation=aggregation)

class Reduce(Metric):

  def __init__(self, reduction, name, dtype=None):
    super(Reduce, self).__init__(name=name, dtype=dtype)
    self.reduction = reduction
    self.total = self.add_weight(
        'total', initializer=init_ops.zeros_initializer)
    if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
                     metrics_utils.Reduction.WEIGHTED_MEAN]:
      self.count = self.add_weight(
          'count', initializer=init_ops.zeros_initializer)

  def update_state(self, values, sample_weight=None):
    values = math_ops.cast(values, self._dtype)
    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)
      values, _, sample_weight = squeeze_or_expand_dimensions(
          values, None, sample_weight)
      try:
        sample_weight = weights_broadcast_ops.broadcast_weights(
            sample_weight, values)
      except ValueError:
        ndim = K.ndim(values)
        weight_ndim = K.ndim(sample_weight)
        if self.reduction == metrics_utils.Reduction.SUM:
          values = math_ops.reduce_sum(
              values, axis=list(range(weight_ndim, ndim)))
        else:
          values = math_ops.reduce_mean(
              values, axis=list(range(weight_ndim, ndim)))
      values = math_ops.multiply(values, sample_weight)

    value_sum = math_ops.reduce_sum(values)
    with ops.control_dependencies([value_sum]):
      update_total_op = self.total.assign_add(value_sum)

    if self.reduction == metrics_utils.Reduction.SUM:
      return update_total_op

    if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
      num_values = math_ops.cast(array_ops.size(values), self._dtype)
    elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
      if sample_weight is None:
        num_values = math_ops.cast(array_ops.size(values), self._dtype)
      else:
        num_values = math_ops.reduce_sum(sample_weight)
    else:
      raise NotImplementedError(
          'reduction [%s] not implemented' % self.reduction)

    with ops.control_dependencies([update_total_op]):
      return self.count.assign_add(num_values)

  def result(self):
    if self.reduction == metrics_utils.Reduction.SUM:
      return array_ops.identity(self.total)
    elif self.reduction in [
        metrics_utils.Reduction.WEIGHTED_MEAN,
        metrics_utils.Reduction.SUM_OVER_BATCH_SIZE
    ]:
      return math_ops.div_no_nan(self.total, self.count)
    else:
      raise NotImplementedError(
          'reduction [%s] not implemented' % self.reduction)

@keras_export('keras.metrics.Sum')
class Sum(Reduce):

  def __init__(self, name='sum', dtype=None):
    super(Sum, self).__init__(reduction=metrics_utils.Reduction.SUM,
                              name=name, dtype=dtype)

@keras_export('keras.metrics.Mean')
class Mean(Reduce):

  def __init__(self, name='mean', dtype=None):
    super(Mean, self).__init__(
        reduction=metrics_utils.Reduction.WEIGHTED_MEAN, name=name, dtype=dtype)

@keras_export('keras.metrics.MeanRelativeError')
class MeanRelativeError(Mean):

  def __init__(self, normalizer, name=None, dtype=None):
    super(MeanRelativeError, self).__init__(name=name, dtype=dtype)
    normalizer = math_ops.cast(normalizer, self._dtype)
    self.normalizer = normalizer

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)

    y_pred, self.normalizer = confusion_matrix.remove_squeezable_dimensions(
        y_pred, self.normalizer)
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    relative_errors = math_ops.div_no_nan(
        math_ops.abs(y_true - y_pred), self.normalizer)

    return super(MeanRelativeError, self).update_state(
        relative_errors, sample_weight=sample_weight)

  def get_config(self):
    n = self.normalizer
    config = {'normalizer': K.eval(n) if is_tensor_or_variable(n) else n}
    base_config = super(MeanRelativeError, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

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
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.Accuracy')
class Accuracy(MeanMetricWrapper):

  def __init__(self, name='accuracy', dtype=None):
    super(Accuracy, self).__init__(accuracy, name, dtype=dtype)

@keras_export('keras.metrics.BinaryAccuracy')
class BinaryAccuracy(MeanMetricWrapper):

  def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
    super(BinaryAccuracy, self).__init__(
        binary_accuracy, name, dtype=dtype, threshold=threshold)

@keras_export('keras.metrics.CategoricalAccuracy')
class CategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='categorical_accuracy', dtype=None):
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)

@keras_export('keras.metrics.SparseCategoricalAccuracy')
class SparseCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)

@keras_export('keras.metrics.TopKCategoricalAccuracy')
class TopKCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
    super(TopKCategoricalAccuracy, self).__init__(
        top_k_categorical_accuracy, name, dtype=dtype, k=k)

@keras_export('keras.metrics.SparseTopKCategoricalAccuracy')
class SparseTopKCategoricalAccuracy(MeanMetricWrapper):

  def __init__(self, k=5, name='sparse_top_k_categorical_accuracy', dtype=None):
    super(SparseTopKCategoricalAccuracy, self).__init__(
        sparse_top_k_categorical_accuracy, name, dtype=dtype, k=k)

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
        {self._confusion_matrix_cond: self.accumulator},
        y_true,
        y_pred,
        thresholds=self.thresholds,
        sample_weight=sample_weight)

  def result(self):
    if len(self.thresholds) == 1:
      result = self.accumulator[0]
    else:
      result = self.accumulator
    return ops.convert_to_tensor(result)

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

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

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
    super(Precision, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_positives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Precision, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.Recall')
class Recall(Metric):

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
    super(Recall, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_negatives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Recall, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@six.add_metaclass(abc.ABCMeta)
class SensitivitySpecificityBase(Metric):

  def __init__(self, value, num_thresholds=200, name=None, dtype=None):
    super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
    if num_thresholds <= 0:
      raise ValueError('`num_thresholds` must be > 0.')
    self.value = value
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.true_negatives = self.add_weight(
        'true_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
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
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        sample_weight=sample_weight)

  def reset_states(self):
    num_thresholds = len(self.thresholds)
    K.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

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
    specificities = math_ops.div_no_nan(
        self.true_negatives, self.true_negatives + self.false_positives)

    min_index = math_ops.argmin(
        math_ops.abs(specificities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(
        self.true_positives[min_index],
        self.true_positives[min_index] + self.false_negatives[min_index])

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
    sensitivities = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_negatives)

    min_index = math_ops.argmin(
        math_ops.abs(sensitivities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    return math_ops.div_no_nan(
        self.true_negatives[min_index],
        self.true_negatives[min_index] + self.false_positives[min_index])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'sensitivity': self.sensitivity
    }
    base_config = super(SpecificityAtSensitivity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.AUC')
class AUC(Metric):

  def __init__(self,
               num_thresholds=200,
               curve='ROC',
               summation_method='interpolation',
               name=None,
               dtype=None,
               thresholds=None):
    if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
        metrics_utils.AUCCurve):
      raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(
          curve, list(metrics_utils.AUCCurve)))
    if isinstance(
        summation_method,
        metrics_utils.AUCSummationMethod) and summation_method not in list(
            metrics_utils.AUCSummationMethod):
      raise ValueError(
          'Invalid summation method: "{}". Valid options are: "{}"'.format(
              summation_method, list(metrics_utils.AUCSummationMethod)))

    if thresholds is not None:
      self.num_thresholds = len(thresholds) + 2
      thresholds = sorted(thresholds)
    else:
      if num_thresholds <= 1:
        raise ValueError('`num_thresholds` must be > 1.')

      self.num_thresholds = num_thresholds
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]

    self.thresholds = [0.0 - K.epsilon()] + thresholds + [1.0 + K.epsilon()]

    if isinstance(curve, metrics_utils.AUCCurve):
      self.curve = curve
    else:
      self.curve = metrics_utils.AUCCurve.from_str(curve)
    if isinstance(summation_method, metrics_utils.AUCSummationMethod):
      self.summation_method = summation_method
    else:
      self.summation_method = metrics_utils.AUCSummationMethod.from_str(
          summation_method)
    super(AUC, self).__init__(name=name, dtype=dtype)

    self.true_positives = self.add_weight(
        'true_positives',
        shape=(self.num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.true_negatives = self.add_weight(
        'true_negatives',
        shape=(self.num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(self.num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(self.num_thresholds,),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
        metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
        metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
        metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
    }, y_true, y_pred, self.thresholds, sample_weight=sample_weight)

  def interpolate_pr_auc(self):
    dtp = self.true_positives[:self.num_thresholds -
                              1] - self.true_positives[1:]
    p = self.true_positives + self.false_positives
    dp = p[:self.num_thresholds - 1] - p[1:]

    prec_slope = math_ops.div_no_nan(
        dtp, math_ops.maximum(dp, 0), name='prec_slope')
    intercept = self.true_positives[1:] - math_ops.multiply(prec_slope, p[1:])

    safe_p_ratio = array_ops.where(
        math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
        math_ops.div_no_nan(
            p[:self.num_thresholds - 1],
            math_ops.maximum(p[1:], 0),
            name='recall_relative_ratio'),
        array_ops.ones_like(p[1:]))

    return math_ops.reduce_sum(
        math_ops.div_no_nan(
            prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
            math_ops.maximum(self.true_positives[1:] + self.false_negatives[1:],
                             0),
            name='pr_auc_increment'),
        name='interpolate_pr_auc')

  def result(self):
    if (self.curve == metrics_utils.AUCCurve.PR and
        self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
       ):
      return self.interpolate_pr_auc()

    recall = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_negatives)
    if self.curve == metrics_utils.AUCCurve.ROC:
      fp_rate = math_ops.div_no_nan(self.false_positives,
                                    self.false_positives + self.true_negatives)
      x = fp_rate
      y = recall
    else:  
      precision = math_ops.div_no_nan(
          self.true_positives, self.true_positives + self.false_positives)
      x = recall
      y = precision

    if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
      heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
    elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
      heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
    else:  
      heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])

    return math_ops.reduce_sum(
        math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights),
        name=self.name)

  def reset_states(self):
    K.batch_set_value(
        [(v, np.zeros((self.num_thresholds,))) for v in self.variables])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'curve': self.curve.value,
        'summation_method': self.summation_method.value,
        'thresholds': self.thresholds[1:-1],
    }
    base_config = super(AUC, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.CosineSimilarity')
class CosineSimilarity(MeanMetricWrapper):

  def __init__(self, name='cosine_similarity', dtype=None, axis=-1):
    super(CosineSimilarity, self).__init__(
        cosine_similarity, name, dtype=dtype, axis=axis)

@keras_export('keras.metrics.MeanAbsoluteError')
class MeanAbsoluteError(MeanMetricWrapper):

  def __init__(self, name='mean_absolute_error', dtype=None):
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name, dtype=dtype)

@keras_export('keras.metrics.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(MeanMetricWrapper):

  def __init__(self, name='mean_absolute_percentage_error', dtype=None):
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name, dtype=dtype)

@keras_export('keras.metrics.MeanSquaredError')
class MeanSquaredError(MeanMetricWrapper):

  def __init__(self, name='mean_squared_error', dtype=None):
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name, dtype=dtype)

@keras_export('keras.metrics.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(MeanMetricWrapper):

  def __init__(self, name='mean_squared_logarithmic_error', dtype=None):
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name, dtype=dtype)

@keras_export('keras.metrics.Hinge')
class Hinge(MeanMetricWrapper):

  def __init__(self, name='hinge', dtype=None):
    super(Hinge, self).__init__(hinge, name, dtype=dtype)

@keras_export('keras.metrics.SquaredHinge')
class SquaredHinge(MeanMetricWrapper):

  def __init__(self, name='squared_hinge', dtype=None):
    super(SquaredHinge, self).__init__(squared_hinge, name, dtype=dtype)

@keras_export('keras.metrics.CategoricalHinge')
class CategoricalHinge(MeanMetricWrapper):

  def __init__(self, name='categorical_hinge', dtype=None):
    super(CategoricalHinge, self).__init__(categorical_hinge, name, dtype=dtype)

@keras_export('keras.metrics.RootMeanSquaredError')
class RootMeanSquaredError(Mean):

  def __init__(self, name='root_mean_squared_error', dtype=None):
    super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)
    error_sq = math_ops.squared_difference(y_pred, y_true)
    return super(RootMeanSquaredError, self).update_state(
        error_sq, sample_weight=sample_weight)

  def result(self):
    return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))

@keras_export('keras.metrics.LogCoshError')
class LogCoshError(MeanMetricWrapper):

  def __init__(self, name='logcosh', dtype=None):
    super(LogCoshError, self).__init__(logcosh, name, dtype=dtype)

@keras_export('keras.metrics.Poisson')
class Poisson(MeanMetricWrapper):

  def __init__(self, name='poisson', dtype=None):
    super(Poisson, self).__init__(poisson, name, dtype=dtype)

@keras_export('keras.metrics.KLDivergence')
class KLDivergence(MeanMetricWrapper):

  def __init__(self, name='kullback_leibler_divergence', dtype=None):
    super(KLDivergence, self).__init__(
        kullback_leibler_divergence, name, dtype=dtype)

@keras_export('keras.metrics.MeanIoU')
class MeanIoU(Metric):

  def __init__(self, num_classes, name=None, dtype=None):
    super(MeanIoU, self).__init__(name=name, dtype=dtype)
    self.num_classes = num_classes

    self.total_cm = self.add_weight(
        'total_confusion_matrix',
        shape=(num_classes, num_classes),
        initializer=init_ops.zeros_initializer,
        dtype=dtypes.float64)

  def update_state(self, y_true, y_pred, sample_weight=None):

    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)

    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])

    if sample_weight is not None and sample_weight.shape.ndims > 1:
      sample_weight = array_ops.reshape(sample_weight, [-1])

    current_cm = confusion_matrix.confusion_matrix(
        y_true,
        y_pred,
        self.num_classes,
        weights=sample_weight,
        dtype=dtypes.float64)
    return self.total_cm.assign_add(current_cm)

  def result(self):
    sum_over_row = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = math_ops.cast(
        array_ops.diag_part(self.total_cm), dtype=self._dtype)

    denominator = sum_over_row + sum_over_col - true_positives

    num_valid_entries = math_ops.reduce_sum(
        math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

    iou = math_ops.div_no_nan(true_positives, denominator)

    return math_ops.div_no_nan(
        math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

  def reset_states(self):
    K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

  def get_config(self):
    config = {'num_classes': self.num_classes}
    base_config = super(MeanIoU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.metrics.MeanTensor')
class MeanTensor(Metric):

  def __init__(self, name='mean_tensor', dtype=None):
    super(MeanTensor, self).__init__(name=name, dtype=dtype)
    self._shape = None
    self._total = None
    self._count = None
    self._built = False

  def _build(self, shape):
    self._shape = tensor_shape.TensorShape(shape)
    self._total = self.add_weight(
        'total', shape=shape, initializer=init_ops.zeros_initializer)
    self._count = self.add_weight(
        'count', shape=shape, initializer=init_ops.zeros_initializer)
    with ops.init_scope():
      if not context.executing_eagerly():
        K._initialize_variables(K._get_session())  
    self._built = True

  @property
  def total(self):
    return self._total if self._built else None

  @property
  def count(self):
    return self._count if self._built else None

  def update_state(self, values, sample_weight=None):
    values = math_ops.cast(values, self._dtype)
    if not self._built:
      self._build(values.shape)
    elif values.shape != self._shape:
      raise ValueError('MeanTensor input values must always have the same '
                       'shape. Expected shape (set during the first call): {}. '
                       'Got: {}'.format(self._shape, values.shape))

    num_values = array_ops.ones_like(values)
    if sample_weight is not None:
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

      num_values = math_ops.multiply(num_values, sample_weight)
      values = math_ops.multiply(values, sample_weight)

    update_total_op = self._total.assign_add(values)
    with ops.control_dependencies([update_total_op]):
      return self._count.assign_add(num_values)

  def result(self):
    if not self._built:
      raise ValueError(
          'MeanTensor does not have any result yet. Please call the MeanTensor '
          'instance or use `.update_state(value)` before retrieving the result.'
          )
    return math_ops.div_no_nan(self.total, self.count)

  def reset_states(self):
    if self._built:
      K.batch_set_value(
          [(v, np.zeros(self._shape.as_list())) for v in self.variables])

@keras_export('keras.metrics.BinaryCrossentropy')
class BinaryCrossentropy(MeanMetricWrapper):

  def __init__(self,
               name='binary_crossentropy',
               dtype=None,
               from_logits=False,
               label_smoothing=0):

    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        label_smoothing=label_smoothing)

@keras_export('keras.metrics.CategoricalCrossentropy')
class CategoricalCrossentropy(MeanMetricWrapper):

  def __init__(self,
               name='categorical_crossentropy',
               dtype=None,
               from_logits=False,
               label_smoothing=0):

    super(CategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        label_smoothing=label_smoothing)

@keras_export('keras.metrics.SparseCategoricalCrossentropy')
class SparseCategoricalCrossentropy(MeanMetricWrapper):

  def __init__(self,
               name='sparse_categorical_crossentropy',
               dtype=None,
               from_logits=False,
               axis=-1):

    super(SparseCategoricalCrossentropy, self).__init__(
        sparse_categorical_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        axis=axis)

class SumOverBatchSize(Reduce):

  def __init__(self, name='sum_over_batch_size', dtype=None):
    super(SumOverBatchSize, self).__init__(
        reduction=metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
        name=name,
        dtype=dtype)

class SumOverBatchSizeMetricWrapper(SumOverBatchSize):

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    super(SumOverBatchSizeMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)

    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(SumOverBatchSizeMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(SumOverBatchSizeMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def accuracy(y_true, y_pred):
  y_pred.shape.assert_is_compatible_with(y_true.shape)
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
  y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
  y_true_rank = ops.convert_to_tensor(y_true).shape.ndims
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])
  y_pred = math_ops.argmax(y_pred, axis=-1)

  if K.dtype(y_pred) != K.dtype(y_true):
    y_pred = math_ops.cast(y_pred, K.dtype(y_true))

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

@keras_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return math_ops.cast(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), K.floatx())

@keras_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
  y_true_rank = ops.convert_to_tensor(y_true).shape.ndims
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])

  return math_ops.cast(
      nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), K.floatx())

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine_proximity = cosine_similarity

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
