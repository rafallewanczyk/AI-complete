import os
import sys
import contextlib
import numpy as np

from .CAInitializer import CAGenerateWeights
import multiprocessing
from joblib import Subprocessor

from utils import std_utils
from .device import device
from interact import interact as io

class nnlib(object):
    device = device 
    DeviceConfig = device.Config
    active_DeviceConfig = DeviceConfig() 

    backend = ""

    dlib = None

    keras = None
    keras_contrib = None

    tf = None
    tf_sess = None

    PML = None
    PMLK = None
    PMLTile= None

    code_import_keras = None
    code_import_keras_contrib = None
    code_import_all = None

    code_import_dlib = None

    ResNet = None
    UNet = None
    UNetTemporalPredictor = None
    NLayerDiscriminator = None

    code_import_keras_string = \

    code_import_keras_contrib_string = \

    code_import_dlib_string = \

    code_import_all_string = \

    @staticmethod
    def _import_tf(device_config):
        if nnlib.tf is not None:
            return nnlib.code_import_tf

        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
            suppressor = std_utils.suppress_stdout_stderr().__enter__()
        else:
            suppressor = None

        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')

        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        import tensorflow as tf
        nnlib.tf = tf

        if device_config.cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()

            if device_config.backend != "tensorflow-generic":
                visible_device_list = ''
                for idx in device_config.gpu_idxs:
                    visible_device_list += str(idx) + ','
                config.gpu_options.visible_device_list=visible_device_list[:-1]

        config.gpu_options.force_gpu_compatible = True
        config.gpu_options.allow_growth = device_config.allow_growth

        nnlib.tf_sess = tf.Session(config=config)

        if suppressor is not None:
            suppressor.__exit__()

    @staticmethod
    def import_keras(device_config):
        if nnlib.keras is not None:
            return nnlib.code_import_keras

        nnlib.backend = device_config.backend
        if "tensorflow" in nnlib.backend:
            nnlib._import_tf(device_config)
        elif nnlib.backend == "plaidML":
            os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
            os.environ["PLAIDML_DEVICE_IDS"] = ",".join ( [ nnlib.device.getDeviceID(idx) for idx in device_config.gpu_idxs] )

        import keras as keras_
        nnlib.keras = keras_

        if 'KERAS_BACKEND' in os.environ:
            os.environ.pop('KERAS_BACKEND')

        if nnlib.backend == "plaidML":
            import plaidml
            import plaidml.tile
            nnlib.PML = plaidml
            nnlib.PMLK = plaidml.keras.backend
            nnlib.PMLTile = plaidml.tile

        if device_config.use_fp16:
            nnlib.keras.backend.set_floatx('float16')

        if "tensorflow" in nnlib.backend:
            nnlib.keras.backend.set_session(nnlib.tf_sess)

        nnlib.keras.backend.set_image_data_format('channels_last')

        nnlib.code_import_keras = compile (nnlib.code_import_keras_string,'','exec')
        nnlib.__initialize_keras_functions()

        return nnlib.code_import_keras

    @staticmethod
    def __initialize_keras_functions():
        keras = nnlib.keras
        K = keras.backend
        KL = keras.layers
        backend = nnlib.backend

        def modelify(model_functor):
            def func(tensor):
                return keras.models.Model (tensor, model_functor(tensor))
            return func

        nnlib.modelify = modelify

        def gaussian_blur(radius=2.0):
            def gaussian(x, mu, sigma):
                return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

            def make_kernel(sigma):
                kernel_size = max(3, int(2 * 2 * sigma + 1))
                mean = np.floor(0.5 * kernel_size)
                kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
                np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
                kernel = np_kernel / np.sum(np_kernel)
                return kernel

            gauss_kernel = make_kernel(radius)
            gauss_kernel = gauss_kernel[:, :,np.newaxis, np.newaxis]

            def func(input):
                inputs = [ input[:,:,:,i:i+1]  for i in range( K.int_shape( input )[-1] ) ]

                outputs = []
                for i in range(len(inputs)):
                    outputs += [ K.conv2d( inputs[i] , K.constant(gauss_kernel) , strides=(1,1), padding="same") ]

                return K.concatenate (outputs, axis=-1)
            return func
        nnlib.gaussian_blur = gaussian_blur

        def style_loss(gaussian_blur_radius=0.0, loss_weight=1.0, wnd_size=0, step_size=1):
            if gaussian_blur_radius > 0.0:
                gblur = gaussian_blur(gaussian_blur_radius)

            def sd(content, style, loss_weight):
                content_nc = K.int_shape(content)[-1]
                style_nc = K.int_shape(style)[-1]
                if content_nc != style_nc:
                    raise Exception("style_loss() content_nc != style_nc")

                axes = [1,2]
                c_mean, c_var = K.mean(content, axis=axes, keepdims=True), K.var(content, axis=axes, keepdims=True)
                s_mean, s_var = K.mean(style, axis=axes, keepdims=True), K.var(style, axis=axes, keepdims=True)
                c_std, s_std = K.sqrt(c_var + 1e-5), K.sqrt(s_var + 1e-5)

                mean_loss = K.sum(K.square(c_mean-s_mean))
                std_loss = K.sum(K.square(c_std-s_std))

                return (mean_loss + std_loss) * ( loss_weight / float(content_nc) )

            def func(target, style):
                if wnd_size == 0:
                    if gaussian_blur_radius > 0.0:
                        return sd( gblur(target), gblur(style), loss_weight=loss_weight)
                    else:
                        return sd( target, style, loss_weight=loss_weight )
                else:
                    if nnlib.tf is not None:
                        sh = K.int_shape(target)[1]
                        k = (sh-wnd_size) // step_size + 1
                        if gaussian_blur_radius > 0.0:
                            target, style = gblur(target), gblur(style)
                        target = nnlib.tf.image.extract_image_patches(target, [1,k,k,1], [1,1,1,1], [1,step_size,step_size,1], 'VALID')
                        style  = nnlib.tf.image.extract_image_patches(style,  [1,k,k,1], [1,1,1,1], [1,step_size,step_size,1], 'VALID')
                        return sd( target, style, loss_weight )
                    if nnlib.PML is not None:
                        print ("Sorry, plaidML backend does not support style_loss")
                        return 0
            return func
        nnlib.style_loss = style_loss

        def dssim(kernel_size=11, k1=0.01, k2=0.03, max_value=1.0):

            def func(y_true, y_pred):
                ch = K.shape(y_pred)[-1]

                def _fspecial_gauss(size, sigma):
                    coords = np.arange(0, size, dtype=K.floatx())
                    coords -= (size - 1 ) / 2.0
                    g = coords**2
                    g *= ( -0.5 / (sigma**2) )
                    g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
                    g = K.constant ( np.reshape (g, (1,-1)) )
                    g = K.softmax(g)
                    g = K.reshape (g, (size, size, 1, 1))
                    g = K.tile (g, (1,1,ch,1))
                    return g

                kernel = _fspecial_gauss(kernel_size,1.5)

                def reducer(x):
                    return K.depthwise_conv2d(x, kernel, strides=(1, 1), padding='valid')

                c1 = (k1 * max_value) ** 2
                c2 = (k2 * max_value) ** 2

                mean0 = reducer(y_true)
                mean1 = reducer(y_pred)
                num0 = mean0 * mean1 * 2.0
                den0 = K.square(mean0) + K.square(mean1)
                luminance = (num0 + c1) / (den0 + c1)

                num1 = reducer(y_true * y_pred) * 2.0
                den1 = reducer(K.square(y_true) + K.square(y_pred))
                c2 *= 1.0 
                cs = (num1 - num0 + c2) / (den1 - den0 + c2)

                ssim_val = K.mean(luminance * cs, axis=(-3, -2) )
                return(1.0 - ssim_val ) / 2.0

            return func

        nnlib.dssim = dssim

        if 'tensorflow' in backend:
            class PixelShuffler(keras.layers.Layer):
                def __init__(self, size=(2, 2),  data_format='channels_last', **kwargs):
                    super(PixelShuffler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):
                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    if self.data_format == 'channels_first':
                        return K.tf.depth_to_space(inputs, self.size[0], 'NCHW')

                    elif self.data_format == 'channels_last':
                        return K.tf.depth_to_space(inputs, self.size[0], 'NHWC')

                def compute_output_shape(self, input_shape):
                    if len(input_shape) != 4:
                        raise ValueError('Inputs should have rank ' +
                                        str(4) +
                                        '; Received input shape:', str(input_shape))

                    if self.data_format == 'channels_first':
                        height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
                        width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
                        channels = input_shape[1] // self.size[0] // self.size[1]

                        if channels * self.size[0] * self.size[1] != input_shape[1]:
                            raise ValueError('channels of input and size are incompatible')

                        return (input_shape[0],
                                channels,
                                height,
                                width)

                    elif self.data_format == 'channels_last':
                        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
                        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
                        channels = input_shape[3] // self.size[0] // self.size[1]

                        if channels * self.size[0] * self.size[1] != input_shape[3]:
                            raise ValueError('channels of input and size are incompatible')

                        return (input_shape[0],
                                height,
                                width,
                                channels)

                def get_config(self):
                    config = {'size': self.size,
                            'data_format': self.data_format}
                    base_config = super(PixelShuffler, self).get_config()

                    return dict(list(base_config.items()) + list(config.items()))
        else:
            class PixelShuffler(KL.Layer):
                def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
                    super(PixelShuffler, self).__init__(**kwargs)
                    self.data_format = data_format
                    self.size = size

                def call(self, inputs):

                    input_shape = K.shape(inputs)
                    if K.int_shape(input_shape)[0] != 4:
                        raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                    if self.data_format == 'channels_first':
                        batch_size, c, h, w = input_shape[0], K.int_shape(inputs)[1], input_shape[2], input_shape[3]
                        rh, rw = self.size
                        oh, ow = h * rh, w * rw
                        oc = c // (rh * rw)

                        out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
                        out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
                        out = K.reshape(out, (batch_size, oc, oh, ow))
                        return out

                    elif self.data_format == 'channels_last':
                        batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], K.int_shape(inputs)[-1]
                        rh, rw = self.size
                        oh, ow = h * rh, w * rw
                        oc = c // (rh * rw)

                        out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
                        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
                        out = K.reshape(out, (batch_size, oh, ow, oc))
                        return out

                def compute_output_shape(self, input_shape):
                    if len(input_shape) != 4:
                        raise ValueError('Inputs should have rank ' +
                                        str(4) +
                                        '; Received input shape:', str(input_shape))

                    if self.data_format == 'channels_first':
                        height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
                        width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
                        channels = input_shape[1] // self.size[0] // self.size[1]

                        if channels * self.size[0] * self.size[1] != input_shape[1]:
                            raise ValueError('channels of input and size are incompatible')

                        return (input_shape[0],
                                channels,
                                height,
                                width)

                    elif self.data_format == 'channels_last':
                        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
                        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
                        channels = input_shape[3] // self.size[0] // self.size[1]

                        if channels * self.size[0] * self.size[1] != input_shape[3]:
                            raise ValueError('channels of input and size are incompatible')

                        return (input_shape[0],
                                height,
                                width,
                                channels)

                def get_config(self):
                    config = {'size': self.size,
                            'data_format': self.data_format}
                    base_config = super(PixelShuffler, self).get_config()

                    return dict(list(base_config.items()) + list(config.items()))

        nnlib.PixelShuffler = PixelShuffler
        nnlib.SubpixelUpscaler = PixelShuffler

        class Scale(KL.Layer):
            def __init__(self, weights=None, axis=-1, gamma_init='zero', **kwargs):
                self.axis = axis
                self.gamma_init = keras.initializers.get(gamma_init)
                self.initial_weights = weights
                super(Scale, self).__init__(**kwargs)

            def build(self, input_shape):
                self.input_spec = [keras.engine.InputSpec(shape=input_shape)]

                self.gamma = K.variable(self.gamma_init((1,)), name='{}_gamma'.format(self.name))
                self.trainable_weights = [self.gamma]

                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights

            def call(self, x, mask=None):
                return self.gamma * x

            def get_config(self):
                config = {"axis": self.axis}
                base_config = super(Scale, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.Scale = Scale

        class Adam(keras.optimizers.Optimizer):

            def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                         epsilon=None, decay=0., amsgrad=False, tf_cpu_mode=0, **kwargs):
                super(Adam, self).__init__(**kwargs)
                with K.name_scope(self.__class__.__name__):
                    self.iterations = K.variable(0, dtype='int64', name='iterations')
                    self.lr = K.variable(lr, name='lr')
                    self.beta_1 = K.variable(beta_1, name='beta_1')
                    self.beta_2 = K.variable(beta_2, name='beta_2')
                    self.decay = K.variable(decay, name='decay')
                if epsilon is None:
                    epsilon = K.epsilon()
                self.epsilon = epsilon
                self.initial_decay = decay
                self.amsgrad = amsgrad
                self.tf_cpu_mode = tf_cpu_mode

            def get_updates(self, loss, params):
                grads = self.get_gradients(loss, params)
                self.updates = [K.update_add(self.iterations, 1)]

                lr = self.lr
                if self.initial_decay > 0:
                    lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                              K.dtype(self.decay))))

                t = K.cast(self.iterations, K.floatx()) + 1
                lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                                   (1. - K.pow(self.beta_1, t)))

                e = K.tf.device("/cpu:0") if self.tf_cpu_mode > 0 else None
                if e: e.__enter__()
                ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                if self.amsgrad:
                    vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
                else:
                    vhats = [K.zeros(1) for _ in params]
                if e: e.__exit__(None, None, None)

                self.weights = [self.iterations] + ms + vs + vhats

                for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
                    e = K.tf.device("/cpu:0") if self.tf_cpu_mode == 2 else None
                    if e: e.__enter__()
                    m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                    v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

                    if self.amsgrad:
                        vhat_t = K.maximum(vhat, v_t)
                        self.updates.append(K.update(vhat, vhat_t))
                    if e: e.__exit__(None, None, None)

                    if self.amsgrad:
                        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                    else:
                        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                    self.updates.append(K.update(m, m_t))
                    self.updates.append(K.update(v, v_t))
                    new_p = p_t

                    if getattr(p, 'constraint', None) is not None:
                        new_p = p.constraint(new_p)

                    self.updates.append(K.update(p, new_p))
                return self.updates

            def get_config(self):
                config = {'lr': float(K.get_value(self.lr)),
                          'beta_1': float(K.get_value(self.beta_1)),
                          'beta_2': float(K.get_value(self.beta_2)),
                          'decay': float(K.get_value(self.decay)),
                          'epsilon': self.epsilon,
                          'amsgrad': self.amsgrad}
                base_config = super(Adam, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.Adam = Adam

        def CAInitializerMP( conv_weights_list ):
            result = CAInitializerMPSubprocessor ( [ (i, K.int_shape(conv_weights)) for i, conv_weights in enumerate(conv_weights_list) ], K.floatx(), K.image_data_format() ).run()
            for idx, weights in result:
                K.set_value ( conv_weights_list[idx], weights )
        nnlib.CAInitializerMP = CAInitializerMP

        if backend == "plaidML":
            class TileOP_ReflectionPadding2D(nnlib.PMLTile.Operation):
                def __init__(self, input, w_pad, h_pad):
                    if K.image_data_format() == 'channels_last':
                        if input.shape.ndims == 4:
                            H, W = input.shape.dims[1:3]
                            if (type(H) == int and h_pad >= H) or \
                                (type(W) == int and w_pad >= W):
                                raise ValueError("Paddings must be less than dimensions.")

                            c = .format(h_pad=h_pad, w_pad=w_pad)
                            if w_pad > 0:
                                c += .format(h_pad=h_pad, w_pad=w_pad)
                            else:
                                c += "LCR = I;"

                            if h_pad > 0:
                                c += .format(h_pad=h_pad, w_pad=w_pad)
                            else:
                                c += "TVB = LCR;"

                            c += "O = TVB; }"

                            inp_dims = input.shape.dims
                            out_dims = (inp_dims[0], inp_dims[1]+h_pad*2, inp_dims[2]+w_pad*2, inp_dims[3])
                        else:
                            raise NotImplemented
                    else:
                        raise NotImplemented

                    super(TileOP_ReflectionPadding2D, self).__init__(c, [('I', input) ],
                            [('O', nnlib.PMLTile.Shape(input.shape.dtype, out_dims ) )])

        class ReflectionPadding2D(keras.layers.Layer):
            def __init__(self, padding=(1, 1), **kwargs):
                self.padding = tuple(padding)
                self.input_spec = [keras.layers.InputSpec(ndim=4)]
                super(ReflectionPadding2D, self).__init__(**kwargs)

            def compute_output_shape(self, s):
                return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

            def call(self, x, mask=None):
                w_pad,h_pad = self.padding
                if "tensorflow" in backend:
                    return K.tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
                elif backend == "plaidML":
                    return TileOP_ReflectionPadding2D.function(x, self.padding[0], self.padding[1])
                else:
                    if K.image_data_format() == 'channels_last':
                        if x.shape.ndims == 4:
                            w = K.concatenate ([ x[:,:,w_pad:0:-1,:],
                                                x,
                                                x[:,:,-2:-w_pad-2:-1,:] ], axis=2 )
                            h = K.concatenate ([ w[:,h_pad:0:-1,:,:],
                                                w,
                                                w[:,-2:-h_pad-2:-1,:,:] ], axis=1 )
                            return h
                        else:
                            raise NotImplemented
                    else:
                        raise NotImplemented

        nnlib.ReflectionPadding2D = ReflectionPadding2D

        class Conv2D():
            def __init__ (self, *args, **kwargs):
                self.reflect_pad = False
                padding = kwargs.get('padding','')
                if padding == 'zero':
                    kwargs['padding'] = 'same'
                if padding == 'reflect':
                    kernel_size = kwargs['kernel_size']
                    if (kernel_size % 2) == 1:
                        self.pad = (kernel_size // 2,)*2
                        kwargs['padding'] = 'valid'
                        self.reflect_pad = True
                self.func = keras.layers.Conv2D (*args, **kwargs)

            def __call__(self,x):
                if self.reflect_pad:
                    x = ReflectionPadding2D( self.pad ) (x)
                return self.func(x)
        nnlib.Conv2D = Conv2D

        class Conv2DTranspose():
            def __init__ (self, *args, **kwargs):
                self.reflect_pad = False
                padding = kwargs.get('padding','')
                if padding == 'zero':
                    kwargs['padding'] = 'same'
                if padding == 'reflect':
                    kernel_size = kwargs['kernel_size']
                    if (kernel_size % 2) == 1:
                        self.pad = (kernel_size // 2,)*2
                        kwargs['padding'] = 'valid'
                        self.reflect_pad = True
                self.func = keras.layers.Conv2DTranspose (*args, **kwargs)

            def __call__(self,x):
                if self.reflect_pad:
                    x = ReflectionPadding2D( self.pad ) (x)
                return self.func(x)
        nnlib.Conv2DTranspose = Conv2DTranspose

    @staticmethod
    def import_keras_contrib(device_config):
        if nnlib.keras_contrib is not None:
            return nnlib.code_import_keras_contrib

        import keras_contrib as keras_contrib_
        nnlib.keras_contrib = keras_contrib_
        nnlib.__initialize_keras_contrib_functions()
        nnlib.code_import_keras_contrib = compile (nnlib.code_import_keras_contrib_string,'','exec')

    @staticmethod
    def __initialize_keras_contrib_functions():
        pass

    @staticmethod
    def import_dlib( device_config = None):
        if nnlib.dlib is not None:
            return nnlib.code_import_dlib

        import dlib as dlib_
        nnlib.dlib = dlib_
        if not device_config.cpu_only and "tensorflow" in device_config.backend and len(device_config.gpu_idxs) > 0:
            nnlib.dlib.cuda.set_device(device_config.gpu_idxs[0])

        nnlib.code_import_dlib = compile (nnlib.code_import_dlib_string,'','exec')

    @staticmethod
    def import_all(device_config = None):
        if nnlib.code_import_all is None:
            if device_config is None:
                device_config = nnlib.active_DeviceConfig
            else:
                nnlib.active_DeviceConfig = device_config

            nnlib.import_keras(device_config)
            nnlib.import_keras_contrib(device_config)
            nnlib.code_import_all = compile (nnlib.code_import_keras_string + '\n'
                                            + nnlib.code_import_keras_contrib_string
                                            + nnlib.code_import_all_string,'','exec')
            nnlib.__initialize_all_functions()

        return nnlib.code_import_all

    @staticmethod
    def __initialize_all_functions():
        exec (nnlib.import_keras(nnlib.active_DeviceConfig), locals(), globals())
        exec (nnlib.import_keras_contrib(nnlib.active_DeviceConfig), locals(), globals())

        class DSSIMMSEMaskLoss(object):
            def __init__(self, mask, is_mse=False):
                self.mask = mask
                self.is_mse = is_mse
            def __call__(self,y_true, y_pred):
                total_loss = None
                mask = self.mask
                if self.is_mse:
                    blur_mask = gaussian_blur(max(1, K.int_shape(mask)[1] // 64))(mask)
                    return K.mean ( 50*K.square( y_true*blur_mask - y_pred*blur_mask ) )
                else:
                    return 10*dssim() (y_true*mask, y_pred*mask)
        nnlib.DSSIMMSEMaskLoss = DSSIMMSEMaskLoss

    @staticmethod
    def finalize_all():
        if nnlib.keras_contrib is not None:
            nnlib.keras_contrib = None

        if nnlib.keras is not None:
            nnlib.keras.backend.clear_session()
            nnlib.keras = None

        if nnlib.tf is not None:
            nnlib.tf_sess = None
            nnlib.tf = None

class CAInitializerMPSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):

        def on_initialize(self, client_dict):
            self.floatx = client_dict['floatx']
            self.data_format = client_dict['data_format']

        def process_data(self, data):
            idx, shape = data
            weights = CAGenerateWeights (shape, self.floatx, self.data_format)
            return idx, weights

        def get_data_name (self, data):
            return "undefined"

    def __init__(self, idx_shapes_list, floatx, data_format ):

        self.idx_shapes_list = idx_shapes_list
        self.floatx = floatx
        self.data_format = data_format

        self.result = []
        super().__init__('CAInitializerMP', CAInitializerMPSubprocessor.Cli)

    def on_clients_initialized(self):
        io.progress_bar ("Initializing CA weights", len (self.idx_shapes_list))

    def on_clients_finalized(self):
        io.progress_bar_close()

    def process_info_generator(self):
        for i in range(multiprocessing.cpu_count()):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'floatx' : self.floatx,
                                      'data_format' : self.data_format
                                      }

    def get_data(self, host_dict):
        if len (self.idx_shapes_list) > 0:
            return self.idx_shapes_list.pop(0)

        return None

    def on_data_return (self, host_dict, data):
        self.idx_shapes_list.insert(0, data)

    def on_result (self, host_dict, data, result):
        self.result.append ( result )
        io.progress_bar_inc(1)

    def get_result(self):
        return self.result
EOF
