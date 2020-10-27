

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

@keras_export('keras.losses.Loss')
class Loss(object):

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, sample_weight=None):
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    with K.name_scope(scope_name or self.__class__.__name__):
      losses = self.call(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    if distribution_strategy_context.has_strategy() and (
        self.reduction == losses_utils.ReductionV2.AUTO or
        self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.reduction.None)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/alpha/tutorials/distribute/training_loops'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction

class LossFunctionWrapper(Loss):

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_error'):
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name=name, reduction=reduction)

@keras_export('keras.losses.MeanAbsoluteError')
class MeanAbsoluteError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_error'):
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name=name, reduction=reduction)

@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_percentage_error'):
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name=name, reduction=reduction)

@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_logarithmic_error'):
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name=name, reduction=reduction)

@keras_export('keras.losses.BinaryCrossentropy')
class BinaryCrossentropy(LossFunctionWrapper):

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)
    self.from_logits = from_logits

@keras_export('keras.losses.CategoricalCrossentropy')
class CategoricalCrossentropy(LossFunctionWrapper):

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_crossentropy'):
    super(CategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)

@keras_export('keras.losses.SparseCategoricalCrossentropy')
class SparseCategoricalCrossentropy(LossFunctionWrapper):

  def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None):
    super(SparseCategoricalCrossentropy, self).__init__(
        sparse_categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits)

@keras_export('keras.losses.Hinge')
class Hinge(LossFunctionWrapper):

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    super(Hinge, self).__init__(hinge, name=name, reduction=reduction)

@keras_export('keras.losses.SquaredHinge')
class SquaredHinge(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='squared_hinge'):
    super(SquaredHinge, self).__init__(
        squared_hinge, name=name, reduction=reduction)

@keras_export('keras.losses.CategoricalHinge')
class CategoricalHinge(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_hinge'):
    super(CategoricalHinge, self).__init__(
        categorical_hinge, name=name, reduction=reduction)

@keras_export('keras.losses.Poisson')
class Poisson(LossFunctionWrapper):

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='poisson'):
    super(Poisson, self).__init__(poisson, name=name, reduction=reduction)

@keras_export('keras.losses.LogCosh')
class LogCosh(LossFunctionWrapper):

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='logcosh'):
    super(LogCosh, self).__init__(logcosh, name=name, reduction=reduction)

@keras_export('keras.losses.KLDivergence')
class KLDivergence(LossFunctionWrapper):

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='kullback_leibler_divergence'):
    super(KLDivergence, self).__init__(
        kullback_leibler_divergence, name=name, reduction=reduction)

@keras_export('keras.losses.Huber')
class Huber(LossFunctionWrapper):

  def __init__(self,
               delta=1.0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='huber_loss'):
    super(Huber, self).__init__(
        huber_loss, name=name, reduction=reduction, delta=delta)

@keras_export('keras.metrics.mean_squared_error',
              'keras.metrics.mse',
              'keras.metrics.MSE',
              'keras.losses.mean_squared_error',
              'keras.losses.mse',
              'keras.losses.MSE')
def mean_squared_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

@keras_export('keras.metrics.mean_absolute_error',
              'keras.metrics.mae',
              'keras.metrics.MAE',
              'keras.losses.mean_absolute_error',
              'keras.losses.mae',
              'keras.losses.MAE')
def mean_absolute_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.abs(y_pred - y_true), axis=-1)

@keras_export('keras.metrics.mean_absolute_percentage_error',
              'keras.metrics.mape',
              'keras.metrics.MAPE',
              'keras.losses.mean_absolute_percentage_error',
              'keras.losses.mape',
              'keras.losses.MAPE')
def mean_absolute_percentage_error(y_true, y_pred):  
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  diff = math_ops.abs(
      (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
  return 100. * K.mean(diff, axis=-1)

@keras_export('keras.metrics.mean_squared_logarithmic_error',
              'keras.metrics.msle',
              'keras.metrics.MSLE',
              'keras.losses.mean_squared_logarithmic_error',
              'keras.losses.msle',
              'keras.losses.MSLE')
def mean_squared_logarithmic_error(y_true, y_pred):  
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
  second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
  return K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)

def _maybe_convert_labels(y_true):
  are_zeros = math_ops.equal(y_true, 0)
  are_ones = math_ops.equal(y_true, 1)
  is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

  def _convert_binary_labels():
    return 2. * y_true - 1.

  updated_y_true = smart_cond.smart_cond(is_binary,
                                         _convert_binary_labels, lambda: y_true)
  return updated_y_true

@keras_export('keras.metrics.squared_hinge', 'keras.losses.squared_hinge')
def squared_hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)

@keras_export('keras.metrics.hinge', 'keras.losses.hinge')
def hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)

@keras_export('keras.losses.categorical_hinge')
def categorical_hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  return math_ops.maximum(0., neg - pos + 1.)

def huber_loss(y_true, y_pred, delta=1.0):
  y_pred = math_ops.cast(y_pred, dtype=K.floatx())
  y_true = math_ops.cast(y_true, dtype=K.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  quadratic = math_ops.minimum(abs_error, delta)
  linear = math_ops.subtract(abs_error, quadratic)
  return math_ops.add(
      math_ops.multiply(
          ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
          math_ops.multiply(quadratic, quadratic)),
      math_ops.multiply(delta, linear))

@keras_export('keras.losses.logcosh')
def logcosh(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)

  def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.log(2.)

  return K.mean(_logcosh(y_pred - y_true), axis=-1)

@keras_export('keras.metrics.categorical_crossentropy',
              'keras.losses.categorical_crossentropy')
def categorical_crossentropy(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

@keras_export('keras.metrics.sparse_categorical_crossentropy',
              'keras.losses.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

@keras_export('keras.metrics.binary_crossentropy',
              'keras.losses.binary_crossentropy')
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):  
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.mean(
      K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)

@keras_export('keras.metrics.kullback_leibler_divergence',
              'keras.metrics.kld',
              'keras.metrics.KLD',
              'keras.losses.kullback_leibler_divergence',
              'keras.losses.kld',
              'keras.losses.KLD')
def kullback_leibler_divergence(y_true, y_pred):  
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)

@keras_export('keras.metrics.poisson', 'keras.losses.poisson')
def poisson(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)

@keras_export(
    'keras.losses.cosine_similarity',
    v1=[
        'keras.metrics.cosine_proximity',
        'keras.metrics.cosine',
        'keras.losses.cosine_proximity',
        'keras.losses.cosine',
        'keras.losses.cosine_similarity',
    ])
def cosine_proximity(y_true, y_pred, axis=-1):
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return math_ops.reduce_sum(y_true * y_pred, axis=axis)

@keras_export('keras.losses.CosineSimilarity')
class CosineSimilarity(LossFunctionWrapper):

  def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='cosine_similarity'):
    super(CosineSimilarity, self).__init__(
        cosine_similarity, reduction=reduction, name=name, axis=axis)

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine_similarity = cosine_proximity

def is_categorical_crossentropy(loss):
  result = ((isinstance(loss, CategoricalCrossentropy) or
             (isinstance(loss, LossFunctionWrapper) and
              loss.fn == categorical_crossentropy) or
             (hasattr(loss, '__name__') and
              loss.__name__ == 'categorical_crossentropy') or
             (loss == 'categorical_crossentropy')))
  return result

@keras_export('keras.losses.serialize')
def serialize(loss):
  return serialize_keras_object(loss)

@keras_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')

@keras_export('keras.losses.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'loss function identifier:', identifier)

LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}
EOF
