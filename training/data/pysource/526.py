

import numpy as np
import numpy.distutils
import numpy.distutils.__config__
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Lambda, Layer, InputSpec, Convolution1D, Convolution2D, add, multiply, Activation, Input, concatenate
from keras.layers.convolutional import _Conv
from keras.layers.merge import _Merge
from keras.layers.recurrent import Recurrent
from keras.utils import conv_utils
from keras.models import Model
from .bn import QuaternionBN as quaternion_normalization
from .bn import sqrt_init
from .init import QuaternionInit
from .norm import LayerNormalization, QuaternionLayerNorm

class QuaternionConv(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 gamma_diag_initializer=sqrt_init,
                 gamma_off_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 init_criterion='he',
                 seed=None,
                 epsilon=1e-7,
                 **kwargs):
        super(QuaternionConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = 'channels_last' if rank == 1 else conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        self.epsilon = epsilon
        if kernel_initializer in ['quaternion', 'quaternion_independent']:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.gamma_diag_initializer = initializers.get(gamma_diag_initializer)
        self.gamma_off_initializer = initializers.get(gamma_off_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis] // 4
        self.kernel_shape = self.kernel_size + (input_dim , self.filters)

        if self.kernel_initializer in {'quaternion', 'quaternion_independent'}:
            kls = {'quaternion': QuaternionInit}[self.kernel_initializer]
            kern_init = kls(
                kernel_size=self.kernel_size,
                input_dim=input_dim,
                weight_dim=self.rank,
                nb_filters=self.filters,
                criterion=self.init_criterion
            )
        else:
            kern_init = self.kernel_initializer
        self.kernel = self.add_weight(
            self.kernel_shape,
            initializer=kern_init,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            bias_shape = (4 * self.filters,)
            self.bias = self.add_weight(
                bias_shape,
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim * 4})
        self.built = True

    def call(self, inputs):
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        input_dim    = K.shape(inputs)[channel_axis] // 4
        if self.rank == 1:
            f_r   = self.kernel[:, :, :self.filters]
            f_i   = self.kernel[:, :, self.filters:self.filters*2]
            f_j   = self.kernel[:, :, self.filters*2:self.filters*3]
            f_k   = self.kernel[:, :, self.filters*3:]
        elif self.rank == 2:
            f_r   = self.kernel[:, :, :, :self.filters]
            f_i   = self.kernel[:, :, :, self.filters:self.filters*2]
            f_j   = self.kernel[:, :, :, self.filters*2:self.filters*3]
            f_k   = self.kernel[:, :, :, self.filters*3:]
        elif self.rank == 3:
            f_r   = self.kernel[:, :, :, :, :self.filters]
            f_i   = self.kernel[:, :, :, :, self.filters:self.filters*2]
            f_j   = self.kernel[:, :, :, :, self.filters*2:self.filters*3]
            f_k   = self.kernel[:, :, :, :, self.filters*3:]

        convArgs = {"strides":       self.strides[0]       if self.rank == 1 else self.strides,
                    "padding":       self.padding,
                    "data_format":   self.data_format,
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate}
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]

        f_r._keras_shape = self.kernel_shape
        f_i._keras_shape = self.kernel_shape
        f_j._keras_shape = self.kernel_shape
        f_k._keras_shape = self.kernel_shape

        cat_kernels_4_r = K.concatenate([f_r, -f_i, -f_j, -f_k], axis=-2)
        cat_kernels_4_i = K.concatenate([f_i,  f_r, -f_k, f_j], axis=-2)
        cat_kernels_4_j = K.concatenate([f_j,  f_k, f_r, -f_i], axis=-2)
        cat_kernels_4_k = K.concatenate([f_k,  -f_j, f_i, f_r], axis=-2)
        cat_kernels_4_quaternion = K.concatenate([cat_kernels_4_r, cat_kernels_4_i, 
                                                  cat_kernels_4_j, cat_kernels_4_k], 
                                                  axis=-1)
        cat_kernels_4_quaternion._keras_shape = self.kernel_size + (4 * input_dim, 4 * self.filters)

        output = convFunc(inputs, cat_kernels_4_quaternion, **convArgs)

        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i]
                )
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (4 * self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (4 * self.filters,) + tuple(new_space)

    def get_config(self):
        if self.kernel_initializer in {'quaternion'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer),
            'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
            'init_criterion': self.init_criterion,
        }
        base_config = super(QuaternionConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class QuaternionConv1D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 **kwargs):
        super(QuaternionConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

class QuaternionConv2D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 **kwargs):
        super(QuaternionConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv2D, self).get_config()
        config.pop('rank')
        return config

class QuaternionConv3D(QuaternionConv):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='quaternion',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 init_criterion='he',
                 **kwargs):
        super(QuaternionConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            **kwargs)

    def get_config(self):
        config = super(QuaternionConv3D, self).get_config()
        config.pop('rank')
        return config

class WeightNorm_Conv(_Conv):

    def __init__(self,
                 gamma_initializer='ones',
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 epsilon=1e-07,
                 **kwargs):
        super(WeightNorm_Conv, self).__init__(**kwargs)
        if self.rank == 1:
            self.data_format = 'channels_last'
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.epsilon = epsilon

    def build(self, input_shape):
        super(WeightNorm_Conv, self).build(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        gamma_shape = (input_dim * self.filters,)
        self.gamma = self.add_weight(
            shape=gamma_shape,
            name='gamma',
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        ker_shape = self.kernel_size + (input_dim, self.filters)
        nb_kernels = ker_shape[-2] * ker_shape[-1]
        kernel_shape_4_norm = (np.prod(self.kernel_size), nb_kernels)
        reshaped_kernel = K.reshape(self.kernel, kernel_shape_4_norm)
        normalized_weight = K.l2_normalize(reshaped_kernel, axis=0, epsilon=self.epsilon)
        normalized_weight = K.reshape(self.gamma, (1, ker_shape[-2] * ker_shape[-1])) * normalized_weight
        shaped_kernel = K.reshape(normalized_weight, ker_shape)
        shaped_kernel._keras_shape = ker_shape
        convArgs = {"strides":       self.strides[0]       if self.rank == 1 else self.strides,
                    "padding":       self.padding,
                    "data_format":   self.data_format,
                    "dilation_rate": self.dilation_rate[0] if self.rank == 1 else self.dilation_rate}
        convFunc = {1: K.conv1d,
                    2: K.conv2d,
                    3: K.conv3d}[self.rank]
        output = convFunc(inputs, shaped_kernel, **convArgs)

        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'epsilon': self.epsilon
        }
        base_config = super(WeightNorm_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

QuaternionConvolution1D = QuaternionConv1D
QuaternionConvolution2D = QuaternionConv2D
QuaternionConvolution3D = QuaternionConv3D
EOF
