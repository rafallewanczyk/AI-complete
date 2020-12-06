

from ast import literal_eval
from keras_applications.xception import Xception
from keras_applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras_applications.inception_v3 import InceptionV3
from keras_applications.resnet_common import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, ResNeXt50, ResNeXt101
from keras_applications.vgg16 import VGG16
from keras_applications.vgg19 import VGG19
from keras_applications.mobilenet import MobileNet
from keras_applications.mobilenet_v2 import MobileNetV2
import keras

import keras_applications
keras_applications.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils
)

def get_regularizer(type, l1, l2):
    if type == 'l1':
        reg = keras.regularizers.l1(l1)
    elif type == 'l2':
        reg = keras.regularizers.l2(l2)
    elif type == 'l1_l2':
        reg = keras.regularizers.l1_l2(l1, l2)
    else:
        reg = None
    return reg

class InputLayer(object):
    def __init__(self, shape):
        self.shape = literal_eval(shape)
        self.keras_layer = keras.layers.Input(shape=self.shape)
        self.type = 'Input'
        self.name = ':'.join([self.type, str(self.shape)])

class ReshapeLayer(object):
    def __init__(self, shape):
        self.shape = literal_eval(shape)
        self.keras_layer = keras.layers.Reshape(target_shape=self.shape)
        self.type = 'Reshape'
        self.name = ':'.join([self.type, str(self.shape)])

class DropoutLayer(object):
    def __init__(self, rate):
        self.rate = literal_eval(rate)
        self.keras_layer = keras.layers.Dropout(rate=self.rate)
        self.type = 'Dropout'
        self.name = ':'.join([self.type, str(self.rate)])

class DenseLayer(object):
    def __init__(self, units):
        self.units = literal_eval(units)
        self.keras_layer = keras.layers.Dense(units=self.units)
        self.type = 'Dense'
        self.name = ':'.join([self.type, str(self.units)])

class ActivationLayer(object):
    def __init__(self, activation):
        self.activation = activation
        self.keras_layer = keras.layers.Activation(activation=self.activation)
        self.type = 'Activation'
        self.name = ':'.join([self.type, self.activation])

class PermuteLayer(object):
    def __init__(self, dims):
        self.dims = dims
        self.keras_layer = keras.layers.Permute(dims=self.dims)
        self.type = 'Permute'
        self.name = ':'.join([self.type, str(self.dims)])

class FlattenLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.Flatten()
        self.type = 'Flatten'
        self.name = 'Flatten'

class SpatialDropout2DLayer(object):
    def __init__(self, rate):
        self.rate = literal_eval(rate)
        self.keras_layer = keras.layers.SpatialDropout2D(rate=self.rate)
        self.type = 'Spatial dropout 2D'
        self.name = ':'.join([self.type, str(self.rate)])

class SpatialDropout3DLayer(object):
    def __init__(self, rate):
        self.rate = literal_eval(rate)
        self.keras_layer = keras.layers.SpatialDropout3D(rate=self.rate)
        self.type = 'Spatial dropout 3D'
        self.name = ':'.join([self.type, str(self.rate)])

class Conv2DLayer(object):
    def __init__(self, maps, kernel, stride, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.Conv2D(self.maps, self.kernel, strides=self.stride,
                                               padding=self.padding, dilation_rate=self.dilation,
                                               kernel_initializer=self.initializer,
                                               kernel_regularizer=self.kernel_regularizer,
                                               activity_regularizer=self.activity_regularizer)
        self.type = 'Convolution 2D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class SeparableConv2DLayer(object):
    def __init__(self, maps, kernel, stride, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.SeparableConv2D(self.maps, self.kernel, strides=self.stride,
                                                        padding=self.padding, dilation_rate=self.dilation,
                                                        kernel_initializer=self.initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        activity_regularizer=self.activity_regularizer)
        self.type = 'Separable convolution 2D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class DepthwiseSeparableConv2DLayer(object):
    def __init__(self, maps, kernel, stride, padding, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.DepthwiseConv2D(self.maps, self.kernel, strides=self.stride,
                                                        padding=self.padding, kernel_initializer=self.initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        activity_regularizer=self.activity_regularizer)
        self.type = 'Depthwise separable convolution 2D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              self.initializer, str(self.kernel_regularizer), str(self.activity_regularizer),
                              self.l1, self.l2])

class ConvTranspose2DLayer(object):
    def __init__(self, maps, kernel, stride, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.Conv2DTranspose(self.maps, self.kernel, strides=self.stride,
                                                        padding=self.padding, dilation_rate=self.dilation,
                                                        kernel_initializer=self.initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        activity_regularizer=self.activity_regularizer)
        self.type = 'Transpose convolution 2D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class ResizeConv2DLayer(object):
    def __init__(self, maps, kernel, stride, upsample, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.upsample = literal_eval(upsample)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_upsample_layer = keras.layers.UpSampling2D(size=self.upsample)
        self.keras_conv_layer = keras.layers.Conv2D(self.maps, self.kernel, strides=self.stride, padding=self.padding,
                                                    dilation_rate=self.dilation, kernel_initializer=self.initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    activity_regularizer=self.activity_regularizer)
        self.type = 'Resize convolution 2D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), str(self.upsample),
                              self.padding, str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class Conv3DLayer(object):
    def __init__(self, maps, kernel, stride, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.Conv3D(self.maps, self.kernel, strides=self.stride, padding=self.padding,
                                               dilation_rate=self.dilation, kernel_initializer=self.initializer,
                                               kernel_regularizer=self.kernel_regularizer,
                                               activity_regularizer=self.activity_regularizer)
        self.type = 'Convolution 3D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class ConvTranspose3DLayer(object):
    def __init__(self, maps, kernel, stride, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_layer = keras.layers.Conv3DTranspose(self.maps, self.kernel, strides=self.stride,
                                                        padding=self.padding, dilation_rate=self.dilation,
                                                        kernel_initializer=self.initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        activity_regularizer=self.activity_regularizer)
        self.type = 'Transpose convolution 3D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), self.padding,
                              str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class ResizeConv3DLayer(object):
    def __init__(self, maps, kernel, stride, upsample, padding, dilation, initializer, kernel_regularizer, activity_regularizer, l1, l2):
        self.maps = literal_eval(maps)
        self.kernel = literal_eval(kernel)
        self.stride = literal_eval(stride)
        self.upsample = literal_eval(upsample)
        self.padding = padding

        if literal_eval(dilation) is None:
            self.dilation = (1, 1)
        else:
            self.dilation = literal_eval(dilation)

        self.initializer = initializer
        self.l1 = l1
        self.l2 = l2
        self.kernel_regularizer = get_regularizer(kernel_regularizer, self.l1, self.l2)
        self.activity_regularizer = get_regularizer(activity_regularizer, self.l1, self.l2)
        self.keras_upsample_layer = keras.layers.UpSampling3D(size=self.upsample)
        self.keras_conv_layer = keras.layers.Conv3D(self.maps, self.kernel, strides=self.stride, padding=self.padding,
                                                    dilation_rate=self.dilation, kernel_initializer=self.initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    activity_regularizer=self.activity_regularizer)
        self.type = 'Resize convolution 3D'
        self.name = ':'.join([self.type, str(self.maps), str(self.kernel), str(self.stride), str(self.upsample),
                              self.padding, str(self.dilation), self.initializer, str(self.kernel_regularizer),
                              str(self.activity_regularizer), self.l1, self.l2])

class Upsample2DLayer(object):
    def __init__(self, size):
        self.size = literal_eval(size)
        self.keras_layer = keras.layers.UpSampling2D(size=self.size)
        self.type = 'Upsample 2D'
        self.name = ':'.join([self.type, str(self.size)])

class Upsample3DLayer(object):
    def __init__(self, size):
        self.size = literal_eval(size)
        self.keras_layer = keras.layers.UpSampling3D(size=self.size)
        self.type = 'Upsample 3D'
        self.name = ':'.join([self.type, str(self.size)])

class ZeroPad2DLayer(object):
    def __init__(self, pad):
        self.pad = literal_eval(pad)
        self.keras_layer = keras.layers.ZeroPadding2D(padding=self.pad)
        self.type = 'Zero padding 2D'
        self.name = ':'.join([self.type, str(self.pad)])

class ZeroPad3DLayer(object):
    def __init__(self, pad):
        self.pad = literal_eval(pad)
        self.keras_layer = keras.layers.ZeroPadding3D(padding=self.pad)
        self.type = 'Zero padding 3D'
        self.name = ':'.join([self.type, str(self.pad)])

class Cropping2DLayer(object):
    def __init__(self, crop):
        self.crop = literal_eval(crop)
        self.keras_layer = keras.layers.Cropping2D(cropping=self.crop)
        self.type = 'Cropping 2D'
        self.name = ':'.join([self.type, str(self.crop)])

class Cropping3DLayer(object):
    def __init__(self, crop):
        self.crop = literal_eval(crop)
        self.keras_layer = keras.layers.Cropping3D(cropping=self.crop)
        self.type = 'Cropping 3D'
        self.name = ':'.join([self.type, str(self.crop)])

class LeakyReluLayer(object):
    def __init__(self, act_param):
        self.type = 'Leaky reLU'

        if act_param:
            self.act_param = literal_eval(act_param)
            self.keras_layer = keras.layers.LeakyReLU(alpha=self.act_param)
            self.name = ':'.join([self.type, str(self.act_param)])
        else:
            self.act_param = act_param
            self.keras_layer = keras.layers.LeakyReLU()
            self.name = ':'.join([self.type, str(0.3)])

class EluLayer(object):
    def __init__(self, act_param):
        self.type = 'ELU'

        if act_param:
            self.act_param = literal_eval(act_param)
            self.keras_layer = keras.layers.ELU(alpha=self.act_param)
            self.name = ':'.join([self.type, str(self.act_param)])
        else:
            self.act_param = act_param
            self.keras_layer = keras.layers.ELU()
            self.name = ':'.join([self.type, str(1.0)])

class ThresholdedReluLayer(object):
    def __init__(self, act_param):
        self.type = 'Thresholded reLU'

        if act_param:
            self.act_param = literal_eval(act_param)
            self.keras_layer = keras.layers.ThresholdedReLU(theta=self.act_param)
            self.name = ':'.join([self.type, str(self.act_param)])
        else:
            self.act_param = act_param
            self.keras_layer = keras.layers.ThresholdedReLU()
            self.name = ':'.join([self.type, str(1.0)])

class PreluLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.PReLU()
        self.type = 'PreLU'
        self.name = 'PreLU'

class MaxPool2DLayer(object):
    def __init__(self, size, stride):
        self.size = literal_eval(size)
        self.stride = literal_eval(stride)
        self.keras_layer = keras.layers.MaxPool2D(pool_size=self.size, strides=self.stride)
        self.type = 'Max pooling 2D'
        self.name = ':'.join([self.type, str(self.size), str(self.stride)])

class AvgPool2DLayer(object):
    def __init__(self, size, stride):
        self.size = literal_eval(size)
        self.stride = literal_eval(stride)
        self.keras_layer = keras.layers.AveragePooling2D(pool_size=self.size, strides=self.stride)
        self.type = 'Average pooling 2D'
        self.name = ':'.join([self.type, str(self.size), str(self.stride)])

class GlobalMaxPool2DLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.GlobalMaxPool2D()
        self.type = 'Global max pooling 2D'
        self.name = 'Global max pooling 2D'

class GlobalAvgPool2DLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.GlobalAvgPool2D()
        self.type = 'Global average pooling 2D'
        self.name = 'Global average pooling 2D'

class MaxPool3DLayer(object):
    def __init__(self, size, stride):
        self.size = literal_eval(size)
        self.stride = literal_eval(stride)
        self.keras_layer = keras.layers.MaxPool3D(pool_size=self.size, strides=self.stride)
        self.type = 'Max pooling 3D'
        self.name = ':'.join([self.type, str(self.size), str(self.stride)])

class AvgPool3DLayer(object):
    def __init__(self, size, stride):
        self.size = literal_eval(size)
        self.stride = literal_eval(stride)
        self.keras_layer = keras.layers.AveragePooling3D(pool_size=self.size, strides=self.stride)
        self.type = 'Average pooling 3D'
        self.name = ':'.join([self.type, str(self.size), str(self.stride)])

class GlobalMaxPool3DLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.GlobalMaxPool3D()
        self.type = 'Global max pooling 3D'
        self.name = 'Global max pooling 3D'

class GlobalAvgPool3DLayer(object):
    def __init__(self):
        self.keras_layer = keras.layers.GlobalAvgPool3D()
        self.type = 'Global average pooling 3D'
        self.name = 'Global average pooling 3D'

class BatchNormalizationLayer(object):
    def __init__(self, momentum, epsilon):
        self.momentum = 0.99
        self.epsilon = 0.001

        if momentum:
            self.momentum = literal_eval(momentum)

        if epsilon:
            self.epsilon = literal_eval(epsilon)

        self.keras_layer = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.type = 'Batch normalization'
        self.name = ':'.join([self.type, str(self.momentum), str(self.epsilon)])

class GaussianDropoutLayer(object):
    def __init__(self, rate):
        self.rate = literal_eval(rate)
        self.keras_layer = keras.layers.GaussianDropout(rate=self.rate)
        self.type = 'Gaussian dropout'
        self.name = ':'.join([self.type, str(self.rate)])

class GaussianNoiseLayer(object):
    def __init__(self, stdev):
        self.stdev = literal_eval(stdev)
        self.keras_layer = keras.layers.GaussianNoise(stddev=self.stdev)
        self.type = 'Gaussian noise'
        self.name = ':'.join([self.type, str(self.stdev)])

class AlphaDropoutLayer(object):
    def __init__(self, rate):
        self.rate = literal_eval(rate)
        self.keras_layer = keras.layers.AlphaDropout(rate=self.rate)
        self.type = 'Alpha dropout'
        self.name = ':'.join([self.type, str(self.rate)])

class OuterSkipConnectionSourceLayer(object):
    def __init__(self, skip_type):
        self.to_merge = []
        self.type = 'Outer skip source'
        self.skip_type = skip_type
        self.name = ':'.join([self.type, self.skip_type])

class OuterSkipConnectionTargetLayer(object):
    def __init__(self, skip_type):
        self.to_merge = []
        self.type = 'Outer skip target'
        self.skip_type = skip_type
        self.name = ':'.join([self.type, self.skip_type])

class InnerSkipConnectionSourceLayer(object):
    def __init__(self, skip_type):
        self.to_merge = []
        self.type = 'Inner skip source'
        self.skip_type = skip_type
        self.name = ':'.join([self.type, self.skip_type])

class InnerSkipConnectionTargetLayer(object):
    def __init__(self, skip_type):
        self.to_merge = []
        self.type = 'Inner skip target'
        self.skip_type = skip_type
        self.name = ':'.join([self.type, self.skip_type])

class XceptionLayer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        inputs = keras.layers.Input(self.input_shape)
        self.keras_layer = Xception(include_top=self.include_top, weights=self.weights, input_tensor=inputs)

        skip1 = self.keras_layer.get_layer('block1_conv1_act').output
        skip1 = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(skip1)
        skip2 = self.keras_layer.get_layer('block3_sepconv2_bn').output
        skip2 = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(skip2)
        skip3 = self.keras_layer.get_layer('block4_sepconv2_bn').output
        skip4 = self.keras_layer.get_layer('block13_sepconv2_bn').output
        skip5 = self.keras_layer.get_layer('block14_sepconv2_act').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'Xception'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class VGG16Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = VGG16(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('block1_conv2').output
        skip2 = self.keras_layer.get_layer('block2_conv2').output
        skip3 = self.keras_layer.get_layer('block3_conv3').output
        skip4 = self.keras_layer.get_layer('block4_conv3').output
        skip5 = self.keras_layer.get_layer('block5_conv3').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'VGG16'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class VGG19Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = VGG19(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('block1_conv2').output
        skip2 = self.keras_layer.get_layer('block2_conv2').output
        skip3 = self.keras_layer.get_layer('block3_conv4').output
        skip4 = self.keras_layer.get_layer('block4_conv4').output
        skip5 = self.keras_layer.get_layer('block5_conv4').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'VGG19'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet50Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet50(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_relu').output
        skip2 = self.keras_layer.get_layer('conv2_block3_out').output
        skip3 = self.keras_layer.get_layer('conv3_block4_out').output
        skip4 = self.keras_layer.get_layer('conv4_block6_out').output
        skip5 = self.keras_layer.get_layer('conv5_block3_out').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet50'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet101Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet101(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_relu').output
        skip2 = self.keras_layer.get_layer('conv2_block3_out').output
        skip3 = self.keras_layer.get_layer('conv3_block4_out').output
        skip4 = self.keras_layer.get_layer('conv4_block23_out').output
        skip5 = self.keras_layer.get_layer('conv5_block3_out').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet101'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet152Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet152(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_relu').output
        skip2 = self.keras_layer.get_layer('conv2_block3_out').output
        skip3 = self.keras_layer.get_layer('conv3_block8_out').output
        skip4 = self.keras_layer.get_layer('conv4_block36_out').output
        skip5 = self.keras_layer.get_layer('conv5_block3_out').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet152'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet50V2Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet50V2(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_conv').output
        skip1 = keras.layers.BatchNormalization()(skip1)
        skip1 = keras.layers.Activation('relu')(skip1)
        skip2 = self.keras_layer.get_layer('conv2_block3_1_relu').output
        skip3 = self.keras_layer.get_layer('conv3_block4_1_relu').output
        skip4 = self.keras_layer.get_layer('conv4_block6_1_relu').output
        skip5 = self.keras_layer.get_layer('post_relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet50V2'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet101V2Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet101V2(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_conv').output
        skip1 = keras.layers.BatchNormalization()(skip1)
        skip1 = keras.layers.Activation('relu')(skip1)
        skip2 = self.keras_layer.get_layer('conv2_block3_1_relu').output
        skip3 = self.keras_layer.get_layer('conv3_block4_1_relu').output
        skip4 = self.keras_layer.get_layer('conv4_block16_1_relu').output
        skip5 = self.keras_layer.get_layer('post_relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet101V2'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNet152V2Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNet152V2(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_conv').output
        skip1 = keras.layers.BatchNormalization()(skip1)
        skip1 = keras.layers.Activation('relu')(skip1)
        skip2 = self.keras_layer.get_layer('conv2_block2_1_relu').output
        skip3 = self.keras_layer.get_layer('conv3_block8_1_relu').output
        skip4 = self.keras_layer.get_layer('conv4_block36_1_relu').output
        skip5 = self.keras_layer.get_layer('post_relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNet152V2'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNeXt50Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNeXt50(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_relu').output
        skip2 = self.keras_layer.get_layer('conv3_block1_1_relu').output
        skip3 = self.keras_layer.get_layer('conv4_block1_1_relu').output
        skip4 = self.keras_layer.get_layer('conv5_block1_1_relu').output
        skip5 = self.keras_layer.get_layer('conv5_block3_out').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNeXt50'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class ResNeXt101Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = ResNeXt101(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1_relu').output
        skip2 = self.keras_layer.get_layer('conv3_block1_1_relu').output
        skip3 = self.keras_layer.get_layer('conv4_block1_1_relu').output
        skip4 = self.keras_layer.get_layer('conv5_block1_1_relu').output
        skip5 = self.keras_layer.get_layer('conv5_block3_out').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'ResNeXt101'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class InceptionV3Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = InceptionV3(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('activation_1').output
        skip1 = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(skip1)
        skip2 = self.keras_layer.get_layer('activation_4').output
        skip2 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip2)
        skip3 = self.keras_layer.get_layer('activation_29').output
        skip3 = keras.layers.ZeroPadding2D(padding=((2, 1), (2, 1)))(skip3)
        skip4 = self.keras_layer.get_layer('activation_75').output
        skip4 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip4)
        skip5 = self.keras_layer.get_layer('mixed10').output
        skip5 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip5)

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'InceptionV3'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class InceptionResNetV2Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = InceptionResNetV2(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('activation_1').output
        skip1 = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(skip1)
        skip2 = self.keras_layer.get_layer('activation_4').output
        skip2 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip2)
        skip3 = self.keras_layer.get_layer('activation_75').output
        skip3 = keras.layers.ZeroPadding2D(padding=((2, 1), (2, 1)))(skip3)
        skip4 = self.keras_layer.get_layer('activation_162').output
        skip4 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip4)
        skip5 = self.keras_layer.get_layer('conv_7b_ac').output
        skip5 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(skip5)

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'InceptionResNetV2'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class DenseNet121Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = DenseNet121(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1/relu').output
        skip2 = self.keras_layer.get_layer('pool2_relu').output
        skip3 = self.keras_layer.get_layer('pool3_relu').output
        skip4 = self.keras_layer.get_layer('pool4_relu').output
        skip5 = self.keras_layer.get_layer('relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'DenseNet121'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class DenseNet169Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = DenseNet169(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1/relu').output
        skip2 = self.keras_layer.get_layer('pool2_relu').output
        skip3 = self.keras_layer.get_layer('pool3_relu').output
        skip4 = self.keras_layer.get_layer('pool4_relu').output
        skip5 = self.keras_layer.get_layer('relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'DenseNet169'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class DenseNet201Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = DenseNet201(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv1/relu').output
        skip2 = self.keras_layer.get_layer('pool2_relu').output
        skip3 = self.keras_layer.get_layer('pool3_relu').output
        skip4 = self.keras_layer.get_layer('pool4_relu').output
        skip5 = self.keras_layer.get_layer('relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'DenseNet201'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class MobileNetLayer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = MobileNet(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('conv_pw_1_relu').output
        skip2 = self.keras_layer.get_layer('conv_pw_3_relu').output
        skip3 = self.keras_layer.get_layer('conv_pw_5_relu').output
        skip4 = self.keras_layer.get_layer('conv_pw_11_relu').output
        skip5 = self.keras_layer.get_layer('conv_pw_13_relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'MobileNet'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class MobileNetV2Layer(object):
    def __init__(self, include_top, weights, input_shape, include_skips, include_hooks):
        self.include_top = literal_eval(include_top)
        self.input_shape = literal_eval(input_shape)
        self.include_skips = include_skips
        self.include_hooks = include_hooks

        if weights == 'none':
            self.weights = None
        else:
            self.weights = weights

        self.keras_layer = MobileNetV2(include_top=self.include_top, weights=self.weights, input_tensor=keras.layers.Input(self.input_shape))

        skip1 = self.keras_layer.get_layer('block_1_expand_relu').output
        skip2 = self.keras_layer.get_layer('block_3_expand_relu').output
        skip3 = self.keras_layer.get_layer('block_6_expand_relu').output
        skip4 = self.keras_layer.get_layer('block_13_expand_relu').output
        skip5 = self.keras_layer.get_layer('out_relu').output

        self.skips = [skip1, skip2, skip3, skip4, skip5]
        self.hooks = [skip1, skip2, skip3, skip4, skip5]

        self.type = 'MobileNetV2'
        self.name = ':'.join([self.type, str(self.include_top), str(self.weights), self.include_skips, self.include_hooks])

class HookConnectionSourceLayer(object):
    def __init__(self):
        self.to_merge = []
        self.type = 'Hook connection source'
        self.name = 'Hook connection source'


EOF
