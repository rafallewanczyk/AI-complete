
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import json
import yaml
import warnings
from six.moves import zip

from .. import backend as K
from .. import optimizers
from ..utils.io_utils import ask_to_proceed_with_overwrite
from ..legacy import models as legacy_models
from ..utils import conv_utils

try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None

def save_model(model, filepath, overwrite=True, include_optimizer=True):

    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    def get_json_type(obj):
        if hasattr(obj, 'get_config'):
            return {'class_name': obj.__class__.__name__,
                    'config': obj.get_config()}

        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return {'type': type(obj),
                        'value': obj.tolist()}
            else:
                return obj.item()

        if callable(obj):
            return obj.__name__

        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError('Not JSON Serializable:', obj)

    from .. import __version__ as keras_version

    if not isinstance(filepath, h5py.File):
        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return

        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False

    try:
        f.attrs['keras_version'] = str(keras_version).encode('utf8')
        f.attrs['backend'] = K.backend().encode('utf8')
        f.attrs['model_config'] = json.dumps({
            'class_name': model.__class__.__name__,
            'config': model.get_config()
        }, default=get_json_type).encode('utf8')

        model_weights_group = f.create_group('model_weights')
        if legacy_models.needs_legacy_support(model):
            model_layers = legacy_models.legacy_sequential_layers(model)
        else:
            model_layers = model.layers
        save_weights_to_hdf5_group(model_weights_group, model_layers)

        if include_optimizer and model.optimizer:
            if isinstance(model.optimizer, optimizers.TFOptimizer):
                warnings.warn(
                    'TensorFlow optimizers do not '
                    'make it possible to access '
                    'optimizer attributes or optimizer state '
                    'after instantiation. '
                    'As a result, we cannot save the optimizer '
                    'as part of the model save file.'
                    'You will have to compile your model again '
                    'after loading it. '
                    'Prefer using a Keras optimizer instead '
                    '(see keras.io/optimizers).')
            else:
                f.attrs['training_config'] = json.dumps({
                    'optimizer_config': {
                        'class_name': model.optimizer.__class__.__name__,
                        'config': model.optimizer.get_config()
                    },
                    'loss': model.loss,
                    'metrics': model.metrics,
                    'sample_weight_mode': model.sample_weight_mode,
                    'loss_weights': model.loss_weights,
                }, default=get_json_type).encode('utf8')

                symbolic_weights = getattr(model.optimizer, 'weights')
                if symbolic_weights:
                    optimizer_weights_group = f.create_group(
                        'optimizer_weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    weight_names = []
                    for i, (w, val) in enumerate(zip(symbolic_weights,
                                                     weight_values)):
                        if K.backend() == 'theano' or K.backend() == 'cntk':
                            if hasattr(w, 'name'):
                                if w.name.split('/')[-1] == 'variable':
                                    name = str(w.name) + '_' + str(i)
                                else:
                                    name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        else:
                            if hasattr(w, 'name') and w.name:
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        weight_names.append(name.encode('utf8'))
                    optimizer_weights_group.attrs[
                        'weight_names'] = weight_names
                    for name, val in zip(weight_names, weight_values):
                        param_dset = optimizer_weights_group.create_dataset(
                            name,
                            val.shape,
                            dtype=val.dtype)
                        if not val.shape:
                            param_dset[()] = val
                        else:
                            param_dset[:] = val
        f.flush()
    finally:
        if opened_new_file:
            f.close()

def load_model(filepath, custom_objects=None, compile=True):
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath

    try:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_config(model_config, custom_objects=custom_objects)

        load_weights_from_hdf5_group(f['model_weights'], model.layers)

        if not compile:
            return model

        training_config = f.attrs.get('training_config')
        if training_config is None:
            warnings.warn('No training configuration found in save file: '
                          'the model was *not* compiled. Compile it manually.')
            return model
        training_config = json.loads(training_config.decode('utf-8'))
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config,
                                           custom_objects=custom_objects)

        loss = convert_custom_objects(training_config['loss'])
        metrics = convert_custom_objects(training_config['metrics'])
        sample_weight_mode = training_config['sample_weight_mode']
        loss_weights = training_config['loss_weights']

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode)

        if 'optimizer_weights' in f:
            if model.__class__.__name__ == 'Sequential':
                model.model._make_train_function()
            else:
                model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [
                n.decode('utf8') for n in
                optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')
        return model
    finally:
        if opened_new_file:
            f.close()

def model_from_config(config, custom_objects=None):
    if isinstance(config, list):
        raise TypeError('`model_from_config` expects a dictionary, '
                        'not a list. Maybe you meant to use '
                        '`Sequential.from_config(config)`?')
    from ..layers import deserialize
    return deserialize(config, custom_objects=custom_objects)

def model_from_yaml(yaml_string, custom_objects=None):
    config = yaml.load(yaml_string)
    from ..layers import deserialize
    return deserialize(config, custom_objects=custom_objects)

def model_from_json(json_string, custom_objects=None):
    config = json.loads(json_string)
    from ..layers import deserialize
    return deserialize(config, custom_objects=custom_objects)

def save_attributes_to_hdf5_group(group, name, data):
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    if len(bad_attributes) > 0:
        raise RuntimeError('The following attributes cannot be saved to HDF5 '
                           'file because they are larger than %d bytes: %s'
                           % (HDF5_OBJECT_HEADER_LIMIT,
                              ', '.join([x for x in bad_attributes])))

    data_npy = np.asarray(data)

    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)

    while any(map(lambda x: x.nbytes > HDF5_OBJECT_HEADER_LIMIT, chunked_data)):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs['%s%d' % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data

def load_attributes_from_hdf5_group(group, name):
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                        for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data

def save_weights_to_hdf5_group(f, layers):
    from .. import __version__ as keras_version

    save_attributes_to_hdf5_group(
        f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
    f.attrs['backend'] = K.backend().encode('utf8')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')

    for layer in layers:
        g = f.create_group(layer.name)
        symbolic_weights = layer.weights
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
        for name, val in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape,
                                          dtype=val.dtype)
            if not val.shape:
                param_dset[()] = val
            else:
                param_dset[:] = val

def preprocess_weights_for_loading(layer, weights,
                                   original_keras_version=None,
                                   original_backend=None,
                                   reshape=False):
    if layer.__class__.__name__ == 'Bidirectional':
        num_weights_per_layer = len(weights) // 2
        forward_weights = preprocess_weights_for_loading(layer.forward_layer,
                                                         weights[:num_weights_per_layer],
                                                         original_keras_version,
                                                         original_backend)
        backward_weights = preprocess_weights_for_loading(layer.backward_layer,
                                                          weights[num_weights_per_layer:],
                                                          original_keras_version,
                                                          original_backend)
        weights = forward_weights + backward_weights

    if original_keras_version == '1':
        if layer.__class__.__name__ == 'TimeDistributed':
            weights = preprocess_weights_for_loading(layer.layer,
                                                     weights,
                                                     original_keras_version,
                                                     original_backend)

        if layer.__class__.__name__ == 'Conv1D':
            shape = weights[0].shape
            if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
                assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0], 1)
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
            weights[0] = weights[0][:, 0, :, :]

        if layer.__class__.__name__ == 'Conv2D':
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

        if layer.__class__.__name__ == 'Conv2DTranspose':
            if layer.data_format == 'channels_last':
                weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

        if layer.__class__.__name__ == 'Conv3D':
            if layer.data_format == 'channels_first':
                weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

        if layer.__class__.__name__ == 'GRU':
            if len(weights) == 9:
                kernel = np.concatenate([weights[0],
                                         weights[3],
                                         weights[6]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[4],
                                                   weights[7]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[5],
                                       weights[8]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'LSTM':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'ConvLSTM2D':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                if layer.data_format == 'channels_first':
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                    recurrent_kernel = np.transpose(recurrent_kernel,
                                                    (2, 3, 1, 0))
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ in ['Model', 'Sequential']:
            new_weights = []
            for sublayer in layer.layers:
                num_weights = len(sublayer.trainable_weights)
                if num_weights > 0:
                    new_weights.extend(preprocess_weights_for_loading(
                        layer=sublayer,
                        weights=weights[:num_weights],
                        original_keras_version=original_keras_version,
                        original_backend=original_backend))
                    weights = weights[num_weights:]

            for sublayer in layer.layers:
                num_weights = len([l for l in sublayer.weights
                                  if l not in sublayer.trainable_weights])
                if num_weights > 0:
                    new_weights.extend(preprocess_weights_for_loading(
                        layer=sublayer,
                        weights=weights[:num_weights],
                        original_keras_version=original_keras_version,
                        original_backend=original_backend))
                    weights = weights[num_weights:]
            weights = new_weights

    conv_layers = ['Conv1D',
                   'Conv2D',
                   'Conv3D',
                   'Conv2DTranspose',
                   'ConvLSTM2D']
    if layer.__class__.__name__ in conv_layers:
        layer_weights_shape = K.int_shape(layer.weights[0])
        if _need_convert_kernel(original_backend):
            weights[0] = conv_utils.convert_kernel(weights[0])
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = conv_utils.convert_kernel(weights[1])
        if reshape and layer_weights_shape != weights[0].shape:
            if weights[0].size != np.prod(layer_weights_shape):
                raise ValueError('Weights must be of equal size to ' +
                                 'apply a reshape operation. ' +
                                 'Layer ' + layer.name +
                                 '\'s weights have shape ' +
                                 str(layer_weights_shape) + ' and size ' +
                                 str(np.prod(layer_weights_shape)) + '. ' +
                                 'The weights for loading have shape ' +
                                 str(weights[0].shape) + ' and size ' +
                                 str(weights[0].size) + '. ')
            weights[0] = np.reshape(weights[0], layer_weights_shape)
        elif layer_weights_shape != weights[0].shape:
            weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

    weights = _convert_rnn_weights(layer, weights)

    return weights

def _convert_rnn_weights(layer, weights):

    def transform_kernels(kernels, func, n_gates):
        return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])

    def transpose_input(from_cudnn):
        order = 'F' if from_cudnn else 'C'

        def transform(kernel):
            return kernel.T.reshape(kernel.shape, order=order)

        return transform

    target_class = layer.__class__.__name__

    if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 4

        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNLSTM'
        elif bias_shape == (units * n_gates,):
            source = 'LSTM'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))

        def convert_weights(weights, from_cudnn=True):
            kernels = transform_kernels(weights[0],
                                        transpose_input(from_cudnn),
                                        n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            if from_cudnn:
                biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
            else:
                biases = np.tile(0.5 * weights[2], 2)
            return [kernels, recurrent_kernels, biases]

        if source != target_class:
            weights = convert_weights(weights, from_cudnn=source == 'CuDNNLSTM')

    if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:

        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 3

        def convert_weights(weights, from_cudnn=True):
            kernels = transform_kernels(weights[0], transpose_input(from_cudnn), n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            biases = weights[2].reshape((2, -1) if from_cudnn else -1)
            return [kernels, recurrent_kernels, biases]

        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNGRU'
        elif bias_shape == (2, units * n_gates):
            source = 'GRU(reset_after=True)'
        elif bias_shape == (units * n_gates,):
            source = 'GRU(reset_after=False)'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))

        if target_class == 'CuDNNGRU':
            target = 'CuDNNGRU'
        elif layer.reset_after:
            target = 'GRU(reset_after=True)'
        else:
            target = 'GRU(reset_after=False)'

        if source != target:
            types = (source, target)
            if 'GRU(reset_after=False)' in types:
                raise ValueError('%s is not compatible with %s' % types)
            if source == 'CuDNNGRU':
                weights = convert_weights(weights, from_cudnn=True)
            elif source == 'GRU(reset_after=True)':
                weights = convert_weights(weights, from_cudnn=False)

    return weights

def _need_convert_kernel(original_backend):
    if original_backend is None:
        return False
    uses_correlation = {'tensorflow': True,
                        'theano': False,
                        'cntk': True}
    return uses_correlation[original_backend] != uses_correlation[K.backend()]

def load_weights_from_hdf5_group(f, layers, reshape=False):
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=reshape)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer 
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)

def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False,
                                         reshape=False):
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
                reshape=reshape)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                  ' due to mismatch in number of weights' +
                                  ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                else:
                    raise ValueError('Layer 
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            for i in range(len(weight_values)):
                if skip_mismatch:
                    if K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                        warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                      ' due to mismatch in shape' +
                                      ' ({} vs {}).'.format(
                                          symbolic_weights[i].shape,
                                          weight_values[i].shape))
                        continue

                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))

    K.batch_set_value(weight_value_tuples)
EOF
