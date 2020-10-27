

import json

import pickle

from abc import abstractmethod

from tensorflow import keras
import numpy as np

IS_CHANNELS_FIRST = keras.backend.image_data_format() == 'channels_first'

class AbstractModelParser:

    def __init__(self, input_model, config):
        self.input_model = input_model
        self.config = config
        self._layer_list = []
        self._layer_dict = {}
        self.parsed_model = None

    def parse(self):

        layers = self.get_layer_iterable()
        snn_layers = eval(self.config.get('restrictions', 'snn_layers'))

        name_map = {}
        idx = 0
        inserted_flatten = False
        for layer in layers:
            layer_type = self.get_type(layer)

            if layer_type == 'BatchNormalization':
                parameters_bn = list(self.get_batchnorm_parameters(layer))
                parameters_bn, axis = parameters_bn[:-1], parameters_bn[-1]
                inbound = self.get_inbound_layers_with_parameters(layer)
                assert len(inbound) == 1, \
                    "Could not find unique layer with parameters " \
                    "preceeding BatchNorm layer."
                prev_layer = inbound[0]
                prev_layer_idx = name_map[str(id(prev_layer))]
                parameters = list(
                    self._layer_list[prev_layer_idx]['parameters'])
                prev_layer_type = self.get_type(prev_layer)
                print("Absorbing batch-normalization parameters into " +
                      "parameters of previous {}.".format(prev_layer_type))

                _depthwise_conv_names = ['DepthwiseConv2D',
                                         'SparseDepthwiseConv2D']
                _sparse_names = ['Sparse', 'SparseConv2D',
                                 'SparseDepthwiseConv2D']
                is_depthwise = prev_layer_type in _depthwise_conv_names
                is_sparse = prev_layer_type in _sparse_names

                if is_sparse:
                    args = [parameters[0], parameters[2]] + parameters_bn
                else:
                    args = parameters[:2] + parameters_bn

                kwargs = {
                    'axis': axis,
                    'image_data_format': keras.backend.image_data_format(),
                    'is_depthwise': is_depthwise}

                params_to_absorb = absorb_bn_parameters(*args, **kwargs)

                if is_sparse:
                    params_to_absorb += (parameters[1],)

                self._layer_list[prev_layer_idx]['parameters'] = \
                    params_to_absorb

            if layer_type == 'GlobalAveragePooling2D':
                print("Replacing GlobalAveragePooling by AveragePooling "
                      "plus Flatten.")
                _layer_type = 'AveragePooling2D'
                axis = 2 if IS_CHANNELS_FIRST else 1
                self._layer_list.append(
                    {'layer_type': _layer_type,
                     'name': self.get_name(layer, idx, _layer_type),
                     'pool_size': (layer.input_shape[axis: axis + 2]),
                     'inbound': self.get_inbound_names(layer, name_map),
                     'strides': [1, 1]})
                name_map[_layer_type + str(idx)] = idx
                idx += 1
                _layer_type = 'Flatten'
                num_str = self.format_layer_idx(idx)
                shape_str = str(np.prod(layer.output_shape[1:]))
                self._layer_list.append(
                    {'name': num_str + _layer_type + '_' + shape_str,
                     'layer_type': _layer_type,
                     'inbound': [self._layer_list[-1]['name']]})
                name_map[_layer_type + str(idx)] = idx
                idx += 1
                inserted_flatten = True

            if layer_type == 'Add':
                print("Replacing Add layer by Concatenate plus Conv.")
                shape = layer.output_shape
                if IS_CHANNELS_FIRST:
                    axis = 1
                    c, h, w = shape[1:]
                    shape_str = '{}x{}x{}'.format(2 * c, h, w)
                else:
                    axis = -1
                    h, w, c = shape[1:]
                    shape_str = '{}x{}x{}'.format(h, w, 2 * c)
                _layer_type = 'Concatenate'
                num_str = self.format_layer_idx(idx)
                self._layer_list.append({
                    'layer_type': _layer_type,
                    'name': num_str + _layer_type + '_' + shape_str,
                    'inbound': self.get_inbound_names(layer, name_map),
                    'axis': axis})
                name_map[_layer_type + str(idx)] = idx
                idx += 1
                _layer_type = 'Conv2D'
                num_str = self.format_layer_idx(idx)
                shape_str = '{}x{}x{}'.format(*shape[1:])
                weights = np.zeros([1, 1, 2 * c, c])
                for k in range(c):
                    weights[:, :, k::c, k] = 1
                self._layer_list.append({
                    'name': num_str + _layer_type + '_' + shape_str,
                    'layer_type': _layer_type,
                    'inbound': [self._layer_list[-1]['name']],
                    'filters': c,
                    'activation': 'relu',  
                    'parameters': (weights, np.zeros(c)),
                    'kernel_size': 1})
                name_map[str(id(layer))] = idx
                idx += 1

            if layer_type not in snn_layers:
                print("Skipping layer {}.".format(layer_type))
                continue

            if not inserted_flatten:
                inserted_flatten = self.try_insert_flatten(layer, idx,
                                                           name_map)
                idx += inserted_flatten

            print("Parsing layer {}.".format(layer_type))

            if layer_type == 'MaxPooling2D' and \
                    self.config.getboolean('conversion', 'max2avg_pool'):
                print("Replacing max by average pooling.")
                layer_type = 'AveragePooling2D'

            if inserted_flatten:
                inbound = [self._layer_list[-1]['name']]
                inserted_flatten = False
            else:
                inbound = self.get_inbound_names(layer, name_map)

            attributes = self.initialize_attributes(layer)

            attributes.update({'layer_type': layer_type,
                               'name': self.get_name(layer, idx),
                               'inbound': inbound})

            if layer_type == 'Dense':
                self.parse_dense(layer, attributes)

            if layer_type == 'Sparse':
                self.parse_sparse(layer, attributes)

            if layer_type in {'Conv1D', 'Conv2D'}:
                self.parse_convolution(layer, attributes)

            if layer_type == 'SparseConv2D':
                self.parse_sparse_convolution(layer, attributes)

            if layer_type == 'DepthwiseConv2D':
                self.parse_depthwiseconvolution(layer, attributes)

            if layer_type == 'SparseDepthwiseConv2D':
                self.parse_sparse_depthwiseconvolution(layer, attributes)

            if layer_type in ['Sparse', 'SparseConv2D',
                              'SparseDepthwiseConv2D']:
                weights, bias, mask = attributes['parameters']

                weights, bias = modify_parameter_precision(
                    weights, bias, self.config, attributes)

                attributes['parameters'] = (weights, bias, mask)

                self.absorb_activation(layer, attributes)

            if layer_type in {'Dense', 'Conv1D', 'Conv2D', 'DepthwiseConv2D'}:
                weights, bias = attributes['parameters']

                weights, bias = modify_parameter_precision(
                    weights, bias, self.config, attributes)

                attributes['parameters'] = (weights, bias)

                self.absorb_activation(layer, attributes)

            if 'Pooling' in layer_type:
                self.parse_pooling(layer, attributes)

            if layer_type == 'Concatenate':
                self.parse_concatenate(layer, attributes)

            self._layer_list.append(attributes)

            name_map[str(id(layer))] = idx

            idx += 1
        print('')

    @abstractmethod
    def get_layer_iterable(self):

        pass

    @abstractmethod
    def get_type(self, layer):

        pass

    @abstractmethod
    def get_batchnorm_parameters(self, layer):

        pass

    def get_inbound_layers_with_parameters(self, layer):

        inbound = layer
        while True:
            inbound = self.get_inbound_layers(inbound)
            if len(inbound) == 1:
                inbound = inbound[0]
                if self.has_weights(inbound):
                    return [inbound]
            else:
                result = []
                for inb in inbound:
                    if self.has_weights(inb):
                        result.append(inb)
                    else:
                        result += self.get_inbound_layers_with_parameters(inb)
                return result

    def get_inbound_names(self, layer, name_map):

        inbound = self.get_inbound_layers(layer)
        for ib in range(len(inbound)):
            for _ in range(len(self.layers_to_skip)):
                if self.get_type(inbound[ib]) in self.layers_to_skip:
                    inbound[ib] = self.get_inbound_layers(inbound[ib])[0]
                else:
                    break
        if len(self._layer_list) == 0 or \
                any([self.get_type(inb) == 'InputLayer' for inb in inbound]):
            return [self.input_layer_name]
        else:
            inb_idxs = [name_map[str(id(inb))] for inb in inbound]
            return [self._layer_list[i]['name'] for i in inb_idxs]

    @abstractmethod
    def get_inbound_layers(self, layer):

        pass

    @property
    def layers_to_skip(self):

        return ['BatchNormalization',
                'Activation',
                'Dropout',
                'ReLU',
                'ActivityRegularization',
                'GaussianNoise']

    @abstractmethod
    def has_weights(self, layer):

        pass

    def initialize_attributes(self, layer=None):

        return {}

    @abstractmethod
    def get_input_shape(self):

        pass

    def get_batch_input_shape(self):

        input_shape = tuple(self.get_input_shape())
        batch_size = self.config.getint('simulation', 'batch_size')
        return (batch_size,) + input_shape

    def get_name(self, layer, idx, layer_type=None):

        if layer_type is None:
            layer_type = self.get_type(layer)

        output_shape = self.get_output_shape(layer)

        shape_string = ["{}x".format(x) for x in output_shape[1:]]
        shape_string[0] = "_" + shape_string[0]
        shape_string[-1] = shape_string[-1][:-1]
        shape_string = "".join(shape_string)

        num_str = self.format_layer_idx(idx)

        return num_str + layer_type + shape_string

    def format_layer_idx(self, idx):

        max_idx = len(self.input_model.layers)
        return str(idx).zfill(len(str(max_idx)))

    @abstractmethod
    def get_output_shape(self, layer):

        pass

    def try_insert_flatten(self, layer, idx, name_map):
        output_shape = self.get_output_shape(layer)
        previous_layers = self.get_inbound_layers(layer)
        prev_layer_output_shape = self.get_output_shape(previous_layers[0])
        if len(output_shape) < len(prev_layer_output_shape) and \
                self.get_type(layer) not in {'Flatten', 'Reshape'} and \
                self.get_type(previous_layers[0]) != 'InputLayer':
            assert len(previous_layers) == 1, \
                "Layer to flatten must be unique."
            print("Inserting layer Flatten.")
            num_str = self.format_layer_idx(idx)
            shape_string = str(np.prod(prev_layer_output_shape[1:]))
            self._layer_list.append({
                'name': num_str + 'Flatten_' + shape_string,
                'layer_type': 'Flatten',
                'inbound': self.get_inbound_names(layer, name_map)})
            name_map['Flatten' + str(idx)] = idx
            return True
        else:
            return False

    @abstractmethod
    def parse_dense(self, layer, attributes):

        pass

    @abstractmethod
    def parse_convolution(self, layer, attributes):

        pass

    @abstractmethod
    def parse_depthwiseconvolution(self, layer, attributes):

        pass

    def parse_sparse(self, layer, attributes):
        pass

    def parse_sparse_convolution(self, layer, attributes):
        pass

    def parse_sparse_depthwiseconvolution(self, layer, attributes):
        pass

    @abstractmethod
    def parse_pooling(self, layer, attributes):

        pass

    def absorb_activation(self, layer, attributes):

        activation_str = self.get_activation(layer)

        outbound = layer
        for _ in range(3):
            outbound = list(self.get_outbound_layers(outbound))
            if len(outbound) != 1:
                break
            else:
                outbound = outbound[0]

                if self.get_type(outbound) == 'Activation':
                    activation_str = self.get_activation(outbound)
                    break

                if self.get_type(outbound) == 'ReLU':
                    print("Parsing ReLU parameters not yet implemented.")
                    activation_str = 'relu'
                    break

                try:
                    self.get_activation(outbound)
                    break
                except AttributeError:
                    pass

        activation, activation_str = get_custom_activation(activation_str)

        if activation_str == 'softmax' and \
                self.config.getboolean('conversion', 'softmax_to_relu'):
            activation = 'relu'
            print("Replaced softmax by relu activation function.")
        elif activation_str == 'linear' and self.get_type(layer) == 'Dense' \
                and self.config.getboolean('conversion', 'append_softmax',
                                           fallback=False):
            activation = 'softmax'
            print("Added softmax.")
        else:
            print("Using activation {}.".format(activation_str))

        attributes['activation'] = activation

    @abstractmethod
    def get_activation(self, layer):

        pass

    @abstractmethod
    def get_outbound_layers(self, layer):

        pass

    @abstractmethod
    def parse_concatenate(self, layer, attributes):

        pass

    def build_parsed_model(self):

        img_input = keras.layers.Input(
            batch_shape=self.get_batch_input_shape(),
            name=self.input_layer_name)
        parsed_layers = {self.input_layer_name: img_input}
        print("Building parsed model...\n")
        for layer in self._layer_list:
            if 'parameters' in layer:
                layer['weights'] = layer.pop('parameters')

            layer_type = layer.pop('layer_type')
            if hasattr(keras.layers, layer_type):
                parsed_layer = getattr(keras.layers, layer_type)
            else:
                import keras_rewiring
                parsed_layer = getattr(keras_rewiring.sparse_layer, layer_type)

            inbound = [parsed_layers[inb] for inb in layer.pop('inbound')]
            if len(inbound) == 1:
                inbound = inbound[0]
            check_for_custom_activations(layer)
            parsed_layers[layer['name']] = parsed_layer(**layer)(inbound)

        print("Compiling parsed model...\n")
        self.parsed_model = keras.models.Model(img_input, parsed_layers[
            self._layer_list[-1]['name']])
        top_k = lambda x, y: keras.metrics.top_k_categorical_accuracy(
            x, y, self.config.getint('simulation', 'top_k'))
        self.parsed_model.compile('sgd', 'categorical_crossentropy',
                                  ['accuracy', top_k])
        self.parsed_model.summary()
        return self.parsed_model

    def evaluate(self, batch_size, num_to_test, x_test=None, y_test=None,
                 dataflow=None):

        assert (x_test is not None and y_test is not None or dataflow is not
                None), "No testsamples provided."

        if x_test is not None:
            score = self.parsed_model.evaluate(x_test, y_test, batch_size,
                                               verbose=0)
        else:
            steps = int(num_to_test / batch_size)
            score = self.parsed_model.evaluate(dataflow, steps=steps)
        print("Top-1 accuracy: {:.2%}".format(score[1]))
        print("Top-5 accuracy: {:.2%}\n".format(score[2]))

        return score

    @property
    def input_layer_name(self):
        return 'input'

def absorb_bn_parameters(weight, bias, mean, var_eps_sqrt_inv, gamma, beta,
                         axis, image_data_format, is_depthwise=False):

    axis = weight.ndim + axis if axis < 0 else axis

    print("Using BatchNorm axis {}.".format(axis))

    if weight.ndim == 4:

        channel_axis = 2 if is_depthwise else 3

        if image_data_format == 'channels_first':
            layer2kernel_axes_map = [None, channel_axis, 0, 1]
        else:
            layer2kernel_axes_map = [None, 0, 1, channel_axis]

        axis = layer2kernel_axes_map[axis]

    broadcast_shape = [1] * weight.ndim
    broadcast_shape[axis] = weight.shape[axis]
    var_eps_sqrt_inv = np.reshape(var_eps_sqrt_inv, broadcast_shape)
    gamma = np.reshape(gamma, broadcast_shape)
    beta = np.reshape(beta, broadcast_shape)
    bias = np.reshape(bias, broadcast_shape)
    mean = np.reshape(mean, broadcast_shape)
    bias_bn = np.ravel(beta + (bias - mean) * gamma * var_eps_sqrt_inv)
    weight_bn = weight * gamma * var_eps_sqrt_inv

    return weight_bn, bias_bn

def modify_parameter_precision(weights, biases, config, attributes):
    if config.getboolean('cell', 'binarize_weights'):
        from snntoolbox.utils.utils import binarize
        print("Binarizing weights.")
        weights = binarize(weights)
    elif config.getboolean('cell', 'quantize_weights'):
        assert 'Qm.f' in attributes, \
            "In the [cell] section of the configuration file, " \
            "'quantize_weights' was set to True. For this to " \
            "work, the layer needs to specify the fixed point " \
            "number format 'Qm.f'."
        from snntoolbox.utils.utils import reduce_precision
        m, f = attributes.get('Qm.f')
        print("Quantizing weights to Q{}.{}.".format(m, f))
        weights = reduce_precision(weights, m, f)
        if attributes.get('quantize_bias', False):
            biases = reduce_precision(biases, m, f)

    attributes.pop('quantize_bias', None)
    attributes.pop('Qm.f', None)

    return weights, biases

def padding_string(pad, pool_size):

    if isinstance(pad, str):
        return pad

    if pad == (0, 0):
        padding = 'valid'
    elif pad == (pool_size[0] // 2, pool_size[1] // 2):
        padding = 'same'
    elif pad == (pool_size[0] - 1, pool_size[1] - 1):
        padding = 'full'
    else:
        raise NotImplementedError(
            "Padding {} could not be interpreted as any of the ".format(pad) +
            "supported border modes 'valid', 'same' or 'full'.")
    return padding

def load_parameters(filepath):

    import h5py

    f = h5py.File(filepath, 'r')

    params = []
    for k in sorted(f.keys()):
        params.append(np.array(f.get(k)))

    f.close()

    return params

def save_parameters(params, filepath, fileformat='h5'):

    if fileformat == 'pkl':
        pickle.dump(params, open(filepath + '.pkl', str('wb')))
    else:
        import h5py
        with h5py.File(filepath, mode='w') as f:
            for i, p in enumerate(params):
                if i < 10:
                    j = '00' + str(i)
                elif i < 100:
                    j = '0' + str(i)
                else:
                    j = str(i)
                f.create_dataset('param_' + j, data=p)

def has_weights(layer):

    return len(layer.weights)

def get_inbound_layers_with_params(layer):

    inbound = layer
    while True:
        inbound = get_inbound_layers(inbound)
        if len(inbound) == 1:
            inbound = inbound[0]
            if has_weights(inbound):
                return [inbound]
        else:
            result = []
            for inb in inbound:
                if has_weights(inb):
                    result.append(inb)
                else:
                    result += get_inbound_layers_with_params(inb)
            return result

def get_inbound_layers_without_params(layer):

    return [layer for layer in get_inbound_layers(layer)
            if not has_weights(layer)]

def get_inbound_layers(layer):

    try:
        inbound_layers = layer._inbound_nodes[0].inbound_layers
    except AttributeError:  
        inbound_layers = layer.inbound_nodes[0].inbound_layers
    if not isinstance(inbound_layers, (list, tuple)):
        inbound_layers = [inbound_layers]
    return inbound_layers

def get_outbound_layers(layer):

    try:
        outbound_nodes = layer._outbound_nodes
    except AttributeError:  
        outbound_nodes = layer.outbound_nodes
    return [on.outbound_layer for on in outbound_nodes]

def get_outbound_activation(layer):

    activation = layer.activation.__name__
    outbound = layer
    for _ in range(2):
        outbound = get_outbound_layers(outbound)
        if len(outbound) == 1 and get_type(outbound[0]) == 'Activation':
            activation = outbound[0].activation.__name__
    return activation

def get_fanin(layer):

    layer_type = get_type(layer)
    if 'Conv' in layer_type:
        ax = 1 if IS_CHANNELS_FIRST else -1
        fanin = np.prod(layer.kernel_size) * layer.input_shape[ax]
    elif 'Dense' in layer_type:
        fanin = layer.input_shape[1]
    elif 'Pool' in layer_type:
        fanin = 0
    else:
        fanin = 0

    return fanin

def get_fanout(layer, config):

    from snntoolbox.simulation.utils import get_spiking_outbound_layers

    next_layers = get_spiking_outbound_layers(layer, config)
    fanout = 0
    for next_layer in next_layers:
        if 'Conv' in next_layer.name and not has_stride_unity(next_layer):
            shape = layer.output_shape
            if 'Input' in get_type(layer):
                shape = fix_input_layer_shape(shape)
            fanout = np.zeros(shape[1:])
            break

    for next_layer in next_layers:
        if 'Dense' in next_layer.name:
            fanout += next_layer.units
        elif 'Pool' in next_layer.name:
            fanout += 1
        elif 'DepthwiseConv' in next_layer.name:
            if has_stride_unity(next_layer):
                fanout += np.prod(next_layer.kernel_size)
            else:
                fanout += get_fanout_array(layer, next_layer, True)
        elif 'Conv' in next_layer.name:
            if has_stride_unity(next_layer):
                fanout += np.prod(next_layer.kernel_size) * next_layer.filters
            else:
                fanout += get_fanout_array(layer, next_layer)

    return fanout

def has_stride_unity(layer):

    return all([s == 1 for s in layer.strides])

def get_fanout_array(layer_pre, layer_post, is_depthwise_conv=False):

    ax = 1 if IS_CHANNELS_FIRST else 0

    nx = layer_post.output_shape[2 + ax]  
    ny = layer_post.output_shape[1 + ax]  
    nz = layer_post.output_shape[ax]  
    kx, ky = layer_post.kernel_size  
    px = int((kx - 1) / 2) if layer_post.padding == 'same' else 0
    py = int((ky - 1) / 2) if layer_post.padding == 'same' else 0
    sx = layer_post.strides[1]
    sy = layer_post.strides[0]

    shape = layer_pre.output_shape
    if 'Input' in get_type(layer_pre):
        shape = fix_input_layer_shape(shape)
    fanout = np.zeros(shape[1:])

    for y_pre in range(fanout.shape[0 + ax]):
        y_post = [int((y_pre + py) / sy)]
        wy = (y_pre + py) % sy
        i = 1
        while wy + i * sy < ky:
            y = y_post[0] - i
            if 0 <= y < ny:
                y_post.append(y)
            i += 1
        for x_pre in range(fanout.shape[1 + ax]):
            x_post = [int((x_pre + px) / sx)]
            wx = (x_pre + px) % sx
            i = 1
            while wx + i * sx < kx:
                x = x_post[0] - i
                if 0 <= x < nx:
                    x_post.append(x)
                i += 1

            if ax:
                fanout[:, y_pre, x_pre] = len(x_post) * len(y_post)
            else:
                fanout[y_pre, x_pre, :] = len(x_post) * len(y_post)

    if not is_depthwise_conv:
        fanout *= nz

    return fanout

def get_type(layer):

    return layer.__class__.__name__

def get_quantized_activation_function_from_string(activation_str):

    from functools import partial
    from snntoolbox.utils.utils import quantized_relu

    m, f = map(int, activation_str[activation_str.index('_Q') + 2:].split('.'))
    activation = partial(quantized_relu, m=m, f=f)
    activation.__name__ = activation_str

    return activation

def get_clamped_relu_from_string(activation_str):

    from snntoolbox.utils.utils import ClampedReLU

    threshold, max_value = map(eval, activation_str.split('_')[-2:])

    activation = ClampedReLU(threshold, max_value)

    return activation

def get_noisy_softplus_from_string(activation_str):
    from snntoolbox.utils.utils import NoisySoftplus

    k, sigma = map(eval, activation_str.split('_')[-2:])

    activation = NoisySoftplus(k, sigma)

    return activation

def get_custom_activation(activation_str):

    if activation_str == 'binary_sigmoid':
        from snntoolbox.utils.utils import binary_sigmoid
        activation = binary_sigmoid
    elif activation_str == 'binary_tanh':
        from snntoolbox.utils.utils import binary_tanh
        activation = binary_tanh
    elif '_Q' in activation_str:
        activation = get_quantized_activation_function_from_string(
            activation_str)
    elif 'clamped_relu' in activation_str:
        activation = get_clamped_relu_from_string(activation_str)
    elif 'NoisySoftplus' in activation_str:
        from snntoolbox.utils.utils import NoisySoftplus
        activation = NoisySoftplus
    else:
        activation = activation_str

    return activation, activation_str

def assemble_custom_dict(*args):
    assembly = []
    for arg in args:
        assembly += arg.items()
    return dict(assembly)

def get_custom_layers_dict(filepath=None):

    from snntoolbox.utils.utils import is_module_installed

    custom_layers = {}
    if is_module_installed('keras_rewiring'):
        from keras_rewiring import Sparse, SparseConv2D, SparseDepthwiseConv2D
        from keras_rewiring.optimizers import NoisySGD

        custom_layers.update({'Sparse': Sparse,
                              'SparseConv2D': SparseConv2D,
                              'SparseDepthwiseConv2D': SparseDepthwiseConv2D,
                              'NoisySGD': NoisySGD})

    if filepath is not None and filepath != '':
        with open(filepath) as f:
            kwargs = json.load(f)
            custom_layers.update(kwargs)

    return custom_layers

def get_custom_activations_dict(filepath=None):

    from snntoolbox.utils.utils import binary_sigmoid, binary_tanh, \
        ClampedReLU, LimitedReLU, NoisySoftplus

    activation_str = 'relu_Q1.4'
    activation = get_quantized_activation_function_from_string(activation_str)

    custom_objects = {
        'binary_sigmoid': binary_sigmoid,
        'binary_tanh': binary_tanh,
        'clamped_relu': ClampedReLU(),
        'LimitedReLU': LimitedReLU,
        'relu6': LimitedReLU({'max_value': 6}),
        activation_str: activation,
        'Noisy_Softplus': NoisySoftplus,
        'precision': precision,
        'activity_regularizer': keras.regularizers.l1}

    if filepath is not None and filepath != '':
        with open(filepath) as f:
            kwargs = json.load(f)

        for key in kwargs:
            if 'LimitedReLU' in key:
                custom_objects[key] = LimitedReLU(kwargs[key])

    return custom_objects

def check_for_custom_activations(layer_attributes):

    if 'activation' not in layer_attributes.keys():
        return

def precision(y_true, y_pred):

    import tensorflow.keras.backend as k
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + k.epsilon())

def fix_input_layer_shape(shape):

    if len(shape) == 1:
        return shape[0]
    return shape
EOF
