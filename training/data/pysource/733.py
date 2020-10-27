

from six import string_types as _string_types

from ...models.neural_network import NeuralNetworkBuilder as _NeuralNetworkBuilder
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from collections import OrderedDict as _OrderedDict
from ...models import datatypes
from ...models import MLModel as _MLModel
from ...models import _MLMODEL_FULL_PRECISION, _MLMODEL_HALF_PRECISION, _VALID_MLMODEL_PRECISION_TYPES
from ...models.utils import convert_neural_network_weights_to_fp16 as convert_neural_network_weights_to_fp16
from ...models.utils import convert_neural_network_spec_weights_to_fp16 as convert_neural_network_spec_weights_to_fp16

from ..._deps import HAS_KERAS_TF as _HAS_KERAS_TF
from ..._deps import HAS_KERAS2_TF as _HAS_KERAS2_TF

if _HAS_KERAS_TF:
    import keras as _keras
    from . import _layers
    from . import _topology
    _KERAS_LAYER_REGISTRY  = {
        _keras.layers.core.Dense: _layers.convert_dense,

        _keras.layers.core.Activation: _layers.convert_activation,
        _keras.layers.advanced_activations.LeakyReLU: _layers.convert_activation,
        _keras.layers.advanced_activations.PReLU: _layers.convert_activation,
        _keras.layers.advanced_activations.ELU: _layers.convert_activation,
        _keras.layers.advanced_activations.ParametricSoftplus: _layers.convert_activation,
        _keras.layers.advanced_activations.ThresholdedReLU: _layers.convert_activation,

        _keras.layers.convolutional.Convolution2D: _layers.convert_convolution,
        _keras.layers.convolutional.Deconvolution2D: _layers.convert_convolution,
        _keras.layers.convolutional.AtrousConvolution2D: _layers.convert_convolution,
        _keras.layers.convolutional.AveragePooling2D: _layers.convert_pooling,
        _keras.layers.convolutional.MaxPooling2D: _layers.convert_pooling,
        _keras.layers.pooling.GlobalAveragePooling2D: _layers.convert_pooling,
        _keras.layers.pooling.GlobalMaxPooling2D: _layers.convert_pooling,
        _keras.layers.convolutional.ZeroPadding2D: _layers.convert_padding,
        _keras.layers.convolutional.Cropping2D: _layers.convert_cropping,
        _keras.layers.convolutional.UpSampling2D: _layers.convert_upsample,

        _keras.layers.convolutional.Convolution1D: _layers.convert_convolution1d,
        _keras.layers.convolutional.AtrousConvolution1D: _layers.convert_convolution1d,
        _keras.layers.convolutional.AveragePooling1D: _layers.convert_pooling,
        _keras.layers.convolutional.MaxPooling1D: _layers.convert_pooling,
        _keras.layers.pooling.GlobalAveragePooling1D: _layers.convert_pooling,
        _keras.layers.pooling.GlobalMaxPooling1D: _layers.convert_pooling,
        _keras.layers.convolutional.ZeroPadding1D: _layers.convert_padding,
        _keras.layers.convolutional.Cropping1D: _layers.convert_cropping,
        _keras.layers.convolutional.UpSampling1D: _layers.convert_upsample,

        _keras.layers.recurrent.LSTM: _layers.convert_lstm,
        _keras.layers.recurrent.SimpleRNN: _layers.convert_simple_rnn,
        _keras.layers.recurrent.GRU: _layers.convert_gru,
        _keras.layers.wrappers.Bidirectional: _layers.convert_bidirectional,

        _keras.layers.normalization.BatchNormalization: _layers.convert_batchnorm,
        _keras.engine.topology.Merge: _layers.convert_merge,
        _keras.layers.core.Flatten: _layers.convert_flatten,
        _keras.layers.core.Permute:_layers.convert_permute,
        _keras.layers.core.Reshape:_layers.convert_reshape,
        _keras.layers.embeddings.Embedding:_layers.convert_embedding,

        _keras.layers.core.RepeatVector:_layers.convert_repeat_vector,

        _keras.engine.topology.InputLayer:_layers.default_skip,
        _keras.layers.core.Dropout:_layers.default_skip,
        _keras.layers.wrappers.TimeDistributed:_layers.default_skip,

    }

    _KERAS_SKIP_LAYERS = [
        _keras.layers.core.Dropout,
    ]

def _check_unsupported_layers(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, _keras.models.Sequential) or isinstance(layer, _keras.models.Model):
            _check_unsupported_layers(layer)
        else: 
            if type(layer) not in _KERAS_LAYER_REGISTRY:
                 raise ValueError(
                     "Keras layer '%s' not supported. " % str(type(layer)))
            if isinstance(layer, _keras.engine.topology.Merge):
                if layer.layers is None: 
                    continue
                for merge_layer in layer.layers:
                    if isinstance(merge_layer, _keras.models.Sequential) or isinstance(merge_layer, _keras.models.Model):
                        _check_unsupported_layers(merge_layer)
            if isinstance(layer, _keras.layers.wrappers.TimeDistributed):
                if type(layer.layer) not in _KERAS_LAYER_REGISTRY:
                     raise ValueError(
                         "Keras layer '%s' not supported. " % str(type(layer.layer)))
            if isinstance(layer, _keras.layers.wrappers.Bidirectional):
                if not isinstance(layer.layer,  _keras.layers.recurrent.LSTM):
                    raise ValueError(
                        "Keras bi-directional wrapper conversion supports only LSTM layer at this time. ")

def _get_layer_converter_fn(layer):
    layer_type = type(layer)
    if layer_type in _KERAS_LAYER_REGISTRY:
        return _KERAS_LAYER_REGISTRY[layer_type]
    else:
        raise TypeError("Keras layer of type %s is not supported." % type(layer))

def _load_keras_model(model_network_path, model_weight_path):
    from keras.models import model_from_json
    import json

    json_file = open(model_network_path, 'r')
    json_string = json_file.read()
    json_file.close()
    loaded_model_json = json.loads(json_string)

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weight_path)

    return loaded_model

def _convert(model, 
            input_names = None, 
            output_names = None, 
            image_input_names = None, 
            is_bgr = False, 
            red_bias = 0.0, 
            green_bias = 0.0, 
            blue_bias = 0.0, 
            gray_bias = 0.0, 
            image_scale = 1.0, 
            class_labels = None, 
            predicted_feature_name = None,
            predicted_probabilities_output = '',
            custom_objects = None):
    if not(_HAS_KERAS_TF):
        raise RuntimeError('keras not found or unsupported version or backend found. keras conversion API is disabled.')
    if isinstance(model, _string_types):
        model = _keras.models.load_model(model, custom_objects = custom_objects)
    elif isinstance(model, tuple):
        model = _load_keras_model(model[0], model[1], custom_objects = custom_objects)
    _check_unsupported_layers(model)
    graph = _topology.NetGraph(model)
    graph.build()
    graph.remove_skip_layers(_KERAS_SKIP_LAYERS)
    graph.insert_1d_permute_layers()
    graph.insert_permute_for_spatial_bn()
    graph.defuse_activation()
    graph.remove_internal_input_layers()
    graph.make_output_layers()

    graph.generate_blob_names()
    graph.add_recurrent_optionals()
    inputs = graph.get_input_layers()
    outputs = graph.get_output_layers()
    if input_names is not None: 
        if isinstance(input_names, _string_types):
            input_names = [input_names]
    else: 
        input_names = ['input' + str(i+1) for i in range(len(inputs))]
    if output_names is not None: 
        if isinstance(output_names, _string_types):
            output_names = [output_names]
    else: 
        output_names = ['output' + str(i+1) for i in range(len(outputs))]
    if image_input_names is not None and isinstance(image_input_names, _string_types):
        image_input_names = [image_input_names]
    graph.reset_model_input_names(input_names)
    graph.reset_model_output_names(output_names)

    if type(model.input_shape) is list:
        input_dims = [list(filter(None, x)) for x in model.input_shape]
        unfiltered_shapes = model.input_shape
    else:
        input_dims = [list(filter(None, model.input_shape))]
        unfiltered_shapes = [model.input_shape]
    for idx, dim in enumerate(input_dims):
        unfiltered_shape = unfiltered_shapes[idx]
        if len(dim) == 0:
            input_dims[idx] = tuple([1])
        elif len(dim) == 1:
            s = graph.get_successors(inputs[idx])[0]
            if isinstance(graph.get_keras_layer(s), _keras.layers.embeddings.Embedding):
                input_dims[idx] = (1,)
            else:
                input_dims[idx] = dim 
        elif len(dim) == 2:  
            input_dims[idx] = (dim[1],)
        elif len(dim) == 3: 
            if (len(unfiltered_shape) > 3):
                input_dims[idx] = (dim[2], dim[0], dim[1])
            else: 
                input_dims[idx] = (dim[2],)
        else:
            raise ValueError('Input' + input_names[idx] + 'has input shape of length' + str(len(dim)))

    if type(model.output_shape) is list:
        output_dims = [list(filter(None, x)) for x in model.output_shape]
    else:
        output_dims = [list(filter(None, model.output_shape[1:]))]

    for idx, dim in enumerate(output_dims):
        if len(dim) == 1:
            output_dims[idx] = dim
        elif len(dim) == 2:  
            output_dims[idx] = (dim[1],)
        elif len(dim) == 3:
            output_dims[idx] = (dim[2], dim[1], dim[0])

    input_types = [datatypes.Array(*dim) for dim in input_dims]
    output_types = [datatypes.Array(*dim) for dim in output_dims]

    input_names = map(str, input_names)
    output_names = map(str, output_names)
    is_classifier = class_labels is not None
    if is_classifier:
        mode = 'classifier'
    else:
        mode = None

    input_features = list(zip(input_names, input_types))
    output_features = list(zip(output_names, output_types))

    builder = _NeuralNetworkBuilder(input_features, output_features, mode = mode)

    for iter, layer in enumerate(graph.layer_list):
        keras_layer = graph.keras_layer_map[layer]
        print("%d : %s, %s" % (iter, layer, keras_layer))
        if isinstance(keras_layer, _keras.layers.wrappers.TimeDistributed):
            keras_layer = keras_layer.layer
        converter_func = _get_layer_converter_fn(keras_layer)
        input_names, output_names = graph.get_layer_blobs(layer)
        converter_func(builder, layer, input_names, output_names, keras_layer)

    builder.set_input(input_names, input_dims)
    builder.set_output(output_names, output_dims)

    builder.add_optionals(graph.optional_inputs, graph.optional_outputs)

    if is_classifier:
        classes_in = class_labels
        if isinstance(classes_in, _string_types):
            import os
            if not os.path.isfile(classes_in):
                raise ValueError("Path to class labels (%s) does not exist." % classes_in)
            with open(classes_in, 'r') as f:
                classes = f.read()
            classes = classes.splitlines()
        elif type(classes_in) is list: 
            classes = classes_in
        else:
            raise ValueError('Class labels must be a list of integers / strings, or a file path')

        if predicted_feature_name is not None:
            builder.set_class_labels(classes, predicted_feature_name = predicted_feature_name,
                                     prediction_blob = predicted_probabilities_output)
        else:
            builder.set_class_labels(classes)

    builder.set_pre_processing_parameters(image_input_names = image_input_names, 
                                          is_bgr = is_bgr, 
                                          red_bias = red_bias, 
                                          green_bias = green_bias, 
                                          blue_bias = blue_bias, 
                                          gray_bias = gray_bias, 
                                          image_scale = image_scale)

    spec = builder.spec
    return spec

def convertToSpec(model,
                  input_names = None,
                  output_names = None,
                  image_input_names = None,
                  is_bgr = False,
                  red_bias = 0.0,
                  green_bias = 0.0,
                  blue_bias = 0.0,
                  gray_bias = 0.0,
                  image_scale = 1.0,
                  class_labels = None,
                  predicted_feature_name = None,
                  model_precision = _MLMODEL_FULL_PRECISION,
                  predicted_probabilities_output = '',
                  add_custom_layers = False,
                  custom_conversion_functions = None,
                  custom_objects=None):

    if model_precision not in _VALID_MLMODEL_PRECISION_TYPES:
        raise RuntimeError('Model precision {} is not valid'.format(model_precision))

    if _HAS_KERAS_TF:
        spec = _convert(model=model,
                         input_names=input_names,
                         output_names=output_names,
                         image_input_names=image_input_names,
                         is_bgr=is_bgr,
                         red_bias=red_bias,
                         green_bias=green_bias,
                         blue_bias=blue_bias,
                         gray_bias=gray_bias,
                         image_scale=image_scale,
                         class_labels=class_labels,
                         predicted_feature_name=predicted_feature_name,
                         predicted_probabilities_output=predicted_probabilities_output,
                         custom_objects=custom_objects)
    elif _HAS_KERAS2_TF:
        from . import _keras2_converter
        spec = _keras2_converter._convert(model=model,
                                           input_names=input_names,
                                           output_names=output_names,
                                           image_input_names=image_input_names,
                                           is_bgr=is_bgr,
                                           red_bias=red_bias,
                                           green_bias=green_bias,
                                           blue_bias=blue_bias,
                                           gray_bias=gray_bias,
                                           image_scale=image_scale,
                                           class_labels=class_labels,
                                           predicted_feature_name=predicted_feature_name,
                                           predicted_probabilities_output=predicted_probabilities_output,
                                           add_custom_layers=add_custom_layers,
                                           custom_conversion_functions=custom_conversion_functions,
                                           custom_objects=custom_objects)
    else:
        raise RuntimeError(
            'Keras not found or unsupported version or backend found. keras conversion API is disabled.')

    if model_precision == _MLMODEL_HALF_PRECISION and model is not None:
        spec = convert_neural_network_spec_weights_to_fp16(spec)

    return spec

def convert(model,
                  input_names = None,
                  output_names = None,
                  image_input_names = None,
                  is_bgr = False,
                  red_bias = 0.0,
                  green_bias = 0.0,
                  blue_bias = 0.0,
                  gray_bias = 0.0,
                  image_scale = 1.0,
                  class_labels = None,
                  predicted_feature_name = None,
                  model_precision = _MLMODEL_FULL_PRECISION,
                  predicted_probabilities_output = '',
                  add_custom_layers = False,
                  custom_conversion_functions = None):
    spec = convertToSpec(model,
                      input_names,
                      output_names,
                      image_input_names,
                      is_bgr,
                      red_bias,
                      green_bias,
                      blue_bias,
                      gray_bias,
                      image_scale,
                      class_labels,
                      predicted_feature_name,
                      model_precision,
                      predicted_probabilities_output,
                      add_custom_layers,
                      custom_conversion_functions=custom_conversion_functions)

    return _MLModel(spec)


EOF
