

import keras
from .. import initializers
from .. import layers
from .. import losses
from keras import backend as K
import numpy as np

import numpy as np

custom_objects = {
    'UpsampleLike'          : layers.UpsampleLike,
    'PriorProbability'      : initializers.PriorProbability,
    'RegressBoxes'          : layers.RegressBoxes,
    'NonMaximumSuppression' : layers.NonMaximumSuppression,
    'Anchors'               : layers.Anchors,
    '_smooth_l1'            : losses.smooth_l1(),
    '_focal'                : losses.focal(),
}

def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    def my_init(shape, dtype=None):
        vec = np.loadtxt('Matlab/VOC_att.txt', dtype='float32', delimiter=',')
        vec = vec[:,:16]
        weight = K.variable(vec,dtype='float32')
        return weight

    outputs = keras.layers.Conv2D(
        filters=64 * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    outputs = keras.layers.Reshape((-1, 64), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Dense(
        units=num_classes,
        kernel_initializer=my_init,
        trainable=False,
    )(outputs)

    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(64, name='pyramid_regression_shafin1', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 64), name='pyramid_regression_shafin2')(outputs)
    def my_init(shape, dtype=None):
        vec = np.loadtxt('Matlab/VOC_att.txt', dtype='float32', delimiter=',')
        vec = vec[:,:16]
        vec = np.expand_dims(vec,0)
        weight = K.variable(vec,dtype='float32')

        return weight

    outputs = keras.layers.Dense(
        units=16,
        kernel_initializer=my_init,
        trainable=False,
    )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def __create_pyramid_features(C3, C4, C5, feature_size=256):
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def default_submodels(num_classes, anchor_parameters):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]

def __build_model_pyramid(name, model, features):
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

def __build_pyramid(models, features):
    return [__build_model_pyramid(n, m, features) for n, m in models]

def __build_anchors(anchor_parameters, features):
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)

def retinanet(
    inputs,
    backbone,
    num_classes,
    anchor_parameters       = AnchorParameters.default,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters)

    _, C3, C4, C5 = backbone.outputs  

    features = create_pyramid_features(C3, C4, C5)

    pyramids = __build_pyramid(submodels, features)
    anchors  = __build_anchors(anchor_parameters, features)

    outputs = [anchors] + pyramids

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def retinanet_bbox(
    inputs,
    num_classes,
    nms        = True,
    name       = 'retinanet-bbox',
    **kwargs
):
    model = retinanet(inputs=inputs, num_classes=num_classes, **kwargs)

    anchors        = model.outputs[0]
    regression     = model.outputs[1]
    classification = model.outputs[2]

    other = model.outputs[3:]

    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])

    if nms:
        detections = layers.NonMaximumSuppression(name='nms')([boxes, classification] + other)
    else:
        detections = keras.layers.Concatenate(axis=2, name='detections')([boxes, classification] + other)

    outputs = [regression, classification] + other + [detections]

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


EOF
