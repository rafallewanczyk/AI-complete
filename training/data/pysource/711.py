import time
import subprocess
import json
import os

import numpy
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0,'extern/libigl/python')
print(os.path.abspath('extern/libigl/python'))
import pyigl as igl
print(igl.__file__)
from utils.iglhelpers import e2p, p2e
from utils import my_utils
from utils.gauss_utils import get_mass_matrix, mass_pca

modes = {0: 'baseline', 1:'pca', 2:'autoencoder'}
current_mode = 0
current_frame = 0
visualize_test_data = True

training_data_root = 'training_data/fixed_material_model/'
test_data_root = 'training_data/fixed_material_model_test/'

def autoencoder_analysis_vae(
    data,
    test_data,
    activation='relu',
    latent_dim=3,
    epochs=100,
    batch_size=100,
    batch_size_increase=False,
    learning_rate=0.001,
    layers=[32, 16],
    pca_weights=None,
    pca_basis=None,
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    callback=None,
    UTMU=None,
    ): 
    assert not((pca_weights is not None) and (pca_basis is not None))  

    import keras
    import keras.backend as K

    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    train_data = data
    test_data = test_data if test_data is not None else data[:10] 

    mean = numpy.mean(train_data)
    std = numpy.std(train_data)
    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)
    print(mean)
    print(std)

    def normalize(data):
        return data
    def denormalize(data):
        return data

    if activation == "my_elu":
        activation = my_utils.create_my_elu()

    def sampling(args):

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    inputs = Input(shape=(len(train_data[0]),), name='encoder_input')
    x = Dense(100, activation=activation)(inputs)
    x = Dense(100, activation=activation)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(latent_dim,), name='encoded_layer')([z_mean, z_log_var])

    vae_encoder = Model(inputs, [z_mean, z_log_var, z], name='vae_encoder')
    vae_encoder.summary()
    encoder = Model(inputs, z_mean, name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(100, activation=activation, name='dense_decode_layer_0')(latent_inputs)
    x = Dense(100, activation=activation, name='dense_decode_layer_1')(x)
    outputs = Dense(len(train_data[0]), activation='linear', name='decoder_output_layer')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(vae_encoder(inputs)[2])
    autoencoder = Model(inputs, outputs, name='vae_mlp')

    from keras.losses import mse, binary_crossentropy

    def vae_loss(inputs, outputs):
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= len(train_data[0])
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return K.mean(reconstruction_loss + kl_loss)
    loss_func = vae_loss

    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss=loss_func,
    )

    hist = History()
    model_start_time = time.time()

    if batch_size_increase:
        n_batch_steps = 5
        max_batch_size = len(train_data)
        min_batch_size = batch_size
        batch_sizes = numpy.linspace(min_batch_size, max_batch_size, n_batch_steps, dtype=int)
        for bs in batch_sizes: 
            autoencoder.fit(
                train_data, train_data,
                epochs=epochs // n_batch_steps,
                batch_size=bs,
                shuffle=True,
                validation_data=(test_data, test_data),
                callbacks=[hist, callback]
                )
    else:
        autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data),
            callbacks=[hist, callback]
            )

    training_time = time.time() - model_start_time
    print("Total training time: ", training_time)

    if model_root:
        print("Saving model...")
        models_with_names = [
            (encoder, "encoder"),
            (decoder, "decoder"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  

            model_json = keras_model.to_json()
            with open(os.path.join(keras_models_dir, name + '.json'), "w") as json_file:
                json_file.write(model_json)
            keras_model.save_weights(os.path.join(keras_models_dir, name + '.h5'))

        print ("Finished saving model.")

    def encode(decoded_data):
        return encoder.predict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(denormalize(decoder.predict(encoded_data)))

    return encode, decode, training_time

def autoencoder_analysis(
    data,
    test_data,
    activation='relu',
    latent_dim=3,
    epochs=100,
    batch_size=100,
    batch_size_increase=False,
    learning_rate=0.001,
    layers=[32, 16],
    pca_weights=None,
    pca_basis=None,
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    callback=None,
    UTMU=None,
    ): 
    assert not((pca_weights is not None) and (pca_basis is not None))  

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    use_validation = False
    if use_validation:
        numpy.random.shuffle(data)
        test_data = data[:len(data)//20]
        train_data = data[len(data)//20:]
    else:
        train_data = data
        test_data = test_data if test_data is not None else data[:10] 

    mean = numpy.mean(train_data)
    std = numpy.std(train_data)
    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)
    print(mean)
    print(std)

    def normalize(data):
        return data
    def denormalize(data):
        return data

    if activation == "my_elu":
        activation = my_utils.create_my_elu()
    input = Input(shape=(len(train_data[0]),), name="encoder_input")
    output = input

    if pca_basis is not None:

        W = pca_basis
        b = numpy.zeros(pca_basis.shape[1])
        output = Dense(pca_basis.shape[1], activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_encode_layer")(output)

    for i, layer_width in enumerate(layers):
        act = activation
        output = Dense(layer_width, activation=act, name="dense_encode_layer_" + str(i))(output)

    output = Dense(latent_dim, activation=activation, name="encoded_layer")(output)  

    for i, layer_width in enumerate(reversed(layers)):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)
    if pca_basis is not None:
        output = Dense(pca_basis.shape[1], activation='linear', name="to_pca_decode_layer")(output) 

        W = pca_basis.T
        b = numpy.zeros(len(train_data[0]))
        output = Dense(len(train_data[0]), activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_decode_layer")(output)
    else:
        output = Dense(len(train_data[0]), activation='linear', name="decoder_output_layer")(output)

    autoencoder = Model(input, output)

    def make_UTMU_loss():
        K_UTMU = K.constant(value=UTMU)
        def UTMU_loss(y_true, y_pred):
            u = y_true - y_pred
            return K.mean(K.dot(u, K.dot(K_UTMU, K.transpose(u))), axis=-1) 

        return UTMU_loss

    loss_func = 'mean_squared_error'
    if UTMU is not None:
        loss_func = make_UTMU_loss()

    lam = 1e-2
    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=autoencoder.get_layer('encoded_layer').get_weights()[0])  
        W = K.transpose(W)  
        h = autoencoder.get_layer('encoded_layer').output
        dh = h * (1 - h)  

        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss=loss_func,
    )

    hist = History()
    model_start_time = time.time()

    in_data = train_data
    out_data = train_data

    denoising = False
    if denoising:
        noise_factor = 0.5
        in_data = train_data + noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=train_data.shape) 

    if batch_size_increase:
        n_batch_steps = 5
        max_batch_size = len(train_data)
        min_batch_size = batch_size
        batch_sizes = numpy.linspace(min_batch_size, max_batch_size, n_batch_steps, dtype=int)
        for bs in batch_sizes: 
            autoencoder.fit(
                in_data, out_data,
                epochs=epochs // n_batch_steps,
                batch_size=bs,
                shuffle=True,
                validation_data=(test_data, test_data),
                callbacks=[hist, callback]
                )
    else:
        autoencoder.fit(
            in_data, out_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data),
            callbacks=[hist, callback]
            )
        do_full_batch_tune = False
        if do_full_batch_tune:
            autoencoder.fit(
                in_data, out_data,
                epochs=1500,
                batch_size=len(train_data),
                shuffle=True,
                validation_data=(test_data, test_data),
                callbacks=[hist, callback]
            )

    training_time = time.time() - model_start_time
    print("Total training time: ", training_time)
    autoencoder,encoder, decoder = my_utils.decompose_ae(autoencoder, do_energy=False)

    if model_root:
        print("Saving model...")
        models_with_names = [
            (autoencoder, "autoencoder"),
            (encoder, "encoder"),
            (decoder, "decoder"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  

            model_json = keras_model.to_json()
            with open(os.path.join(keras_models_dir, name + '.json'), "w") as json_file:
                json_file.write(model_json)
            keras_model.save_weights(os.path.join(keras_models_dir, name + '.h5'))

        print ("Finished saving model.")

    def encode(decoded_data):
        return encoder.predict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(denormalize(decoder.predict(encoded_data)))

    return encode, decode, training_time

def distance_errors(samples, reencoded):
    per_vert_error_vec = (samples - reencoded).reshape(len(samples[0])//3*len(samples),3)
    dist_errors = numpy.sum(numpy.abs(per_vert_error_vec)**2,axis=-1)**(1./2)
    return dist_errors

def pca_with_error_cutoff(samples, max_allowable_error, components=None, mem_=None):
    from scipy import linalg

    print('Doing SVD on', len(samples), 'samples')

    if components is None:
        _, S, components = linalg.svd(samples, full_matrices=False)

    def get_for_dim(pca_dim):
        U = components[:pca_dim].T

        explained_variance_ratio = 0

        def encode(samples):
            return samples @ U

        def decode(samples):
            return samples @ U.T

        dist_errors = distance_errors(samples, decode(encode(samples)))
        max_error = numpy.max(dist_errors)
        return max_error, U, explained_variance_ratio, encode, decode

    dim_list = list(reversed(range(1, len(samples[0]))))

    mem = mem_ if mem_ is not None else {}
    def bisect_left():
        lo = len(dim_list) - 300 
        hi = len(dim_list)
        while lo < hi:
            mid = (lo+hi)//2

            if dim_list[mid] not in mem:
                mem[dim_list[mid]] = get_for_dim(dim_list[mid])
            print(dim_list[mid], mem[dim_list[mid]][0])
            if mem[dim_list[mid]][0] < max_allowable_error:
                lo = mid+1
            else:
                hi = mid
        return mem[dim_list[lo - 1]] if dim_list[lo - 1] in mem else get_for_dim(dim_list[lo - 1])

    max_err, U, explained_variance_ratio, encode, decode = bisect_left()
    print("PCA basis of size", len(U[0]), " has max distance error of", max_err, "<", max_allowable_error)
    return U, explained_variance_ratio, encode, decode, components, mem

def pca_no_centering(samples, pca_dim, components=None):
    from scipy import linalg

    print('Doing PCA for', pca_dim, 'components...')

    if components is None:
        _, S, components = linalg.svd(samples, full_matrices=False)
    U = components[:pca_dim].T

    explained_variance_ = 0
    total_var = 0 
    explained_variance_ratio = 0 

    def encode(samples):
        return samples @ U

    def decode(samples):
        return samples @ U.T

    return U, explained_variance_ratio, encode, decode

def mass_pca_analysis(samples, pca_dim, mesh_path, density):
    U = mass_pca(mesh_path, density, samples, pca_dim, eng=None)  

    def encode(samples):
        return samples @ U

    def decode(samples):
        return samples @ U.T

    return U, 0.0, encode, decode

def generate_model(
    model_root, 
    config, 
    ):
    learning_config = config['learning_config']
    autoencoder_config = learning_config['autoencoder_config']
    energy_model_config = learning_config['energy_model_config']
    save_objs = learning_config['save_objs']
    train_in_full_space = autoencoder_config['train_in_full_space']
    use_pca_init = autoencoder_config['pca_init']
    record_full_mse_each_epoch = learning_config['record_full_loss']
    assert train_in_full_space if not use_pca_init else True 

    training_data_path = config['training_dataset']
    validation_data_path = ''
    mesh_path = os.path.join(model_root, 'tets.mesh')

    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(mesh_path)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
    if os.path.exists(validation_data_path):
        test_displacements = my_utils.load_displacement_dmats_to_numpy(validation_data_path)
    else:
        test_displacements = None
    flatten_data, unflatten_data = my_utils.get_flattners(displacements)
    displacements = flatten_data(displacements)
    test_displacements = flatten_data(test_displacements)

    use_mass_pca = learning_config.get("use_mass_pca")
    density = 1.0
    with open(os.path.join(training_data_path, 'parameters.json')) as f:
        density = json.load(f)['density']

    use_UTMU_metric = learning_config.get("use_UTMU_metric")
    if use_UTMU_metric or use_mass_pca:
        mass_matrix = get_mass_matrix(mesh_path, density)

    outer_dim_max_vert_error = autoencoder_config.get('outer_dim_max_vert_error', None) or autoencoder_config['pca_layer_err']
    inner_dim_max_vert_error = autoencoder_config.get('inner_dim_max_vert_error', None) or autoencoder_config['pca_layer_err']
    ae_encoded_dim_min = autoencoder_config['ae_encoded_dim_min']
    ae_encoded_dim_max = autoencoder_config['ae_encoded_dim_max']
    ae_epochs = autoencoder_config['training_epochs']
    batch_size = autoencoder_config['batch_size'] if autoencoder_config['batch_size'] > 0 else len(displacements)
    batch_size_increase = autoencoder_config.get('batch_size_increase', False)
    learning_rate = autoencoder_config['learning_rate']
    layers = autoencoder_config['non_pca_layer_sizes']
    do_fine_tuning = autoencoder_config['do_fine_tuning']
    activation = autoencoder_config['activation']
    save_pca_components = True
    save_autoencoder = True

    pca_start_time = time.time()
    U_ae, explained_var, high_dim_pca_encode, high_dim_pca_decode, components, mem = pca_with_error_cutoff(displacements, outer_dim_max_vert_error)
    pca_train_time = time.time() - pca_start_time

    if train_in_full_space:
        high_dim_pca_encode = lambda x: x
        high_dim_pca_decode = lambda x: x
    encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)

    if test_displacements is not None:
        encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)
    else:
        encoded_high_dim_pca_test_displacements = None

    from keras.callbacks import Callback 
    class MyHistoryCallback(Callback):

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            if record_full_mse_each_epoch:
                mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_displacements)), displacements)
                if test_displacements is not None:
                    val_mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_test_displacements)), test_displacements)
                else:
                    val_mse = 0
                self.history.setdefault('mse', []).append(mse)
                self.history.setdefault('val_mse', []).append(val_mse)
                print()
                print("Mean squared error: ", mse)
                print("Val Mean squared error: ", val_mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    ae_pca_basis_path = os.path.join(model_root, 'pca_results/ae_pca_components.dmat')
    print('Saving pca results to', ae_pca_basis_path)
    my_utils.create_dir_if_not_exist(os.path.dirname(ae_pca_basis_path))
    my_utils.save_numpy_mat_to_dmat(ae_pca_basis_path, numpy.ascontiguousarray(U_ae))

    training_results = {"autoencoder": {}, "pca": {}}
    chosen_dim = ae_encoded_dim_max
    for ae_dim in range(ae_encoded_dim_min, ae_encoded_dim_max + 1):
        this_training_results = {}
        pca_basis_init = U_ae if train_in_full_space and use_pca_init else None
        ae_encode, ae_decode, ae_train_time = autoencoder_analysis(
                                        encoded_high_dim_pca_displacements,
                                        encoded_high_dim_pca_test_displacements,
                                        activation=activation,
                                        latent_dim=ae_dim,
                                        epochs=ae_epochs,
                                        batch_size=batch_size,
                                        batch_size_increase=batch_size_increase,
                                        learning_rate=learning_rate,
                                        layers=layers, 
                                        pca_basis=pca_basis_init,
                                        do_fine_tuning=do_fine_tuning,
                                        model_root=model_root,
                                        autoencoder_config=autoencoder_config,
                                        callback=my_hist_callback,
                                        UTMU=None,
                                    )

        decoded_autoencoder_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(displacements))))
        if test_displacements is not None:
            decoded_autoencoder_test_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(test_displacements))))
        ae_training_mse = mean_squared_error(flatten_data(decoded_autoencoder_displacements), flatten_data(displacements))
        ae_relative_percent_error = compute_relative_percent_error(flatten_data(decoded_autoencoder_displacements), flatten_data(displacements))
        dist_errors = distance_errors(decoded_autoencoder_displacements, displacements)
        ae_max_dist_error = numpy.max(dist_errors)
        ae_mean_dist_error = numpy.mean(dist_errors)
        this_training_results['training-mse'] = ae_training_mse
        this_training_results['relative_percent_error'] = ae_relative_percent_error
        this_training_results['max_distance_error'] = ae_max_dist_error
        this_training_results['mean_distance_error'] = ae_mean_dist_error
        this_training_results['ae_training_time_s'] = ae_train_time
        this_training_results['outer_pca_training_time_s'] = pca_train_time
        if test_displacements is not None:
            this_training_results['validation-mse'] = mean_squared_error(flatten_data(decoded_autoencoder_test_displacements), flatten_data(test_displacements))

        print('Finale ae training MSE:', ae_training_mse)
        print('Finale ae relative percent error:', ae_relative_percent_error)
        print('Finale ae max distance error:', ae_max_dist_error)
        print('Finale ae mean distance error:', ae_mean_dist_error)

        training_results['autoencoder']["inner dim " + str(ae_dim)] = this_training_results
        if ae_max_dist_error < inner_dim_max_vert_error:
            chosen_dim = ae_dim
            break

    U_best_match, _, _, _, _, _ = pca_with_error_cutoff(displacements, ae_max_dist_error, components, mem)

    outer_layer_dim = U_ae.shape[1]
    best_match_pca_dim = U_best_match.shape[1]
    ae_encoded_dim = chosen_dim

    pca_compare_dims = [outer_layer_dim, best_match_pca_dim, ae_encoded_dim]

    for pca_dim in pca_compare_dims:
        U, explained_var, pca_encode, pca_decode = pca_no_centering(displacements, pca_dim, components)

        encoded_pca_displacements = pca_encode(displacements)
        decoded_pca_displacements = pca_decode(encoded_pca_displacements)
        if test_displacements is not None:
            encoded_pca_test_displacements = pca_encode(test_displacements)
            decoded_pca_test_displacements = pca_decode(encoded_pca_test_displacements)

        if save_pca_components:
            pca_results_filename = os.path.join(model_root, 'pca_results/pca_components_' + str(pca_dim) + '.dmat')
            print('Saving pca results to', pca_results_filename)
            my_utils.create_dir_if_not_exist(os.path.dirname(pca_results_filename))
            my_utils.save_numpy_mat_to_dmat(pca_results_filename, numpy.ascontiguousarray(U))

            training_mse = mean_squared_error(flatten_data(decoded_pca_displacements), flatten_data(displacements))

            actual = flatten_data(displacements)
            pred = flatten_data(decoded_pca_displacements)
            relative_error = compute_relative_percent_error(pred, actual)
            dist_errors = distance_errors(decoded_pca_displacements, displacements)
            max_distance_error = numpy.max(dist_errors)
            mean_distance_error = numpy.mean(dist_errors)

            if test_displacements is not None:
                validation_mse = mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements))
            training_results['pca'][str(pca_dim) + '-components'] = {}
            training_results['pca'][str(pca_dim) + '-components']['training-mse'] = training_mse
            training_results['pca'][str(pca_dim) + '-components']['max-distance-error'] = max_distance_error
            training_results['pca'][str(pca_dim) + '-components']['mean-distance-error'] = mean_distance_error

            if test_displacements is not None:
                training_results['pca'][str(pca_dim) + '-components']['validation-mse'] = validation_mse

            print(str(pca_dim) + ' training MSE: ', training_mse)
            print(str(pca_dim) + ' training worst distance error: ', max_distance_error)
            print(str(pca_dim) + ' training mean distance error: ', mean_distance_error)
            print()

    with open(os.path.join(model_root, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=2)

    history_path = os.path.join(model_root, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(my_hist_callback.history, f, indent=2)

    if save_objs:
        print("Saving objs of decoded training data...")
        obj_dir = os.path.join(model_root, 'decoded_training_objs/')
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        for i, dec_displ in enumerate(decoded_autoencoder_displacements):
            decoded_verts = base_verts + dec_displ.reshape((len(base_verts), 3))
            decoded_verts_eig = p2e(decoded_verts)

            obj_path = os.path.join(obj_dir, "decoded_%05d.obj" % i)
            igl.writeOBJ(obj_path, decoded_verts_eig, face_indices_eig)

    return outer_layer_dim, best_match_pca_dim, ae_encoded_dim

def main():
    sparse_learn()

def load_displacements_and_energy(model_root, use_reencoded=False, use_extra_samples=False):
    from keras.models import Model, load_model
    encoder = load_model(os.path.join(model_root,'keras_models/encoder.hdf5'))
    U = igl.eigen.MatrixXd()
    igl.readDMAT(os.path.join(model_root, 'pca_results/ae_pca_components.dmat'), U)

    if not use_reencoded and not use_extra_samples:
        training_data_path = os.path.join(model_root,'training_data/training')
        displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
        flatten_displ, unflatten_displ = my_utils.get_flattners(displacements)
        encoded_displacements = encoder.predict(flatten_displ(displacements) @ U)
        energies = my_utils.load_energy_dmats_to_numpy(training_data_path)

    if use_reencoded:
        reencoded_training_data_path = os.path.join(model_root, 'augmented_training_data/reencoded/')
        displacements_path = os.path.join(reencoded_training_data_path, 'displacements.dmat')
        enc_displacements_path = os.path.join(reencoded_training_data_path, 'enc_displacements.dmat')
        energies_path = os.path.join(reencoded_training_data_path, 'energies.dmat')
        displacements = my_utils.read_double_dmat_to_numpy(displacements_path)
        encoded_displacements = my_utils.read_double_dmat_to_numpy(enc_displacements_path)
        energies = my_utils.read_double_dmat_to_numpy(energies_path)

    if use_extra_samples:
        sampled_training_data_path = os.path.join(model_root, 'augmented_training_data/sampled/')
        displacements_path = os.path.join(sampled_training_data_path, 'displacements.dmat')
        enc_displacements_path = os.path.join(sampled_training_data_path, 'enc_displacements.dmat')
        energies_path = os.path.join(sampled_training_data_path, 'energies.dmat')
        displacements = numpy.append(displacements, my_utils.read_double_dmat_to_numpy(displacements_path), axis=0)
        encoded_displacements = numpy.append(encoded_displacements, my_utils.read_double_dmat_to_numpy(enc_displacements_path), axis=0)
        energies = numpy.append(energies, my_utils.read_double_dmat_to_numpy(energies_path), axis=0)

    return displacements, encoded_displacements, energies

def build_energy_model(model_root, config):
    energy_model_config = config['learning_config']['energy_model_config']
    base_path = model_root
    training_data_path = config['training_dataset']
    validation_data_path = ""
    energies = my_utils.load_energy_dmats_to_numpy(training_data_path)
    if os.path.exists(validation_data_path):
        energies_test = my_utils.load_energy_dmats_to_numpy(validation_data_path)
    else:
        energies_test = None
    flatten_data, unflatten_data = my_utils.get_flattners(energies)

    energies = flatten_data(energies)
    energies_test = flatten_data(energies_test)

    if not os.path.exists(os.path.join(base_path, 'pca_results/')):
        os.makedirs(os.path.join(base_path, 'pca_results/'))
    basis_path = os.path.join(base_path, 'pca_results/energy_pca_components.dmat')
    tet_path = os.path.join(base_path, 'pca_results/energy_indices.dmat')
    start = time.time()
    basis_opt(
        energy_model_config,
        model_root,
        basis_path,
        tet_path,
        energies,
        energies_test,
        energy_model_config['pca_dim'],
        energy_model_config['num_sample_tets'],
        True,
        target_mins=energy_model_config['target_anneal_mins'],
        brute_force_its=energy_model_config['brute_force_iterations']
    )
    print("Energy optimization took", time.time() - start)

def basis_opt(energy_model_config, model_root, basis_output_path, index_output_path, samples, samples_test, pca_dim, num_tets, do_save, target_mins=0.5, brute_force_its=100):
    start = time.time()
    n_tets_sampled = num_tets
    U, explained_variance_ratio, encode, decode = pca_no_centering(samples, _with_error_cutoffpca_dim)

    if do_save:
        my_utils.save_numpy_mat_to_dmat(basis_output_path, numpy.ascontiguousarray(U))

    decoded_energy = decode(encode(samples))
    if samples_test is not None:
        decoded_test_energy = decode(encode(samples_test))

    pca_train_mse = mean_squared_error(samples, decoded_energy)
    print('train mse: ', pca_train_mse)    
    if samples_test is not None:
        print('test mse:', mean_squared_error(samples_test, decoded_test_energy))    
    print('explained var', sum(explained_variance_ratio))
    Es = samples 
    Es_summed = numpy.sum(Es, axis=1)

    UTU = U.transpose() @ U
    UTE = U.transpose() @ Es.T
    U_sum = numpy.sum(U, axis=0)
    training_log = {'best_energy': []}

    n_random = brute_force_its
    print("\nDoing", n_random, "random iterations...")
    min_mse = 1e100
    best_sample = None
    for i in range(n_random):
        if((i + 1) % 100 == 0):
            print("Iteration:", i + 1)
        tet_sample = numpy.random.choice(U.shape[0], num_tets, replace=False)

        U_bar = U[tet_sample, :]
        E_bars = Es[:, tet_sample]

        start_solve = time.time()
        if True:
            sol = numpy.linalg.lstsq(U_bar, E_bars.T)
            alphas = sol[0]
            start = time.time()
            completion_mse = 1.0/(Es.shape[0] * Es.shape[1]) * (numpy.trace(alphas.transpose() @ UTU @ alphas) - numpy.trace(alphas.transpose() @ UTE)) 
        else:
            sol = numpy.linalg.solve(U_bar.T @ U_bar, U_bar.T)
            E_stars = (U_sum.T @ sol) @ E_bars.T
            completion_mse = mean_squared_error(E_stars, Es_summed) 

        if completion_mse < min_mse:
            min_mse = completion_mse
            best_sample = tet_sample
            print('min mse (minus constant):', min_mse)

        training_log['best_energy'].append(min_mse)
    a = best_sample

    from simanneal import Annealer
    class Problem(Annealer):
        def __init__(self, state, min_mse):
            super(Problem, self).__init__(state)  
            self.min_mse = min_mse

        def move(self):

            for _ in range(1):
                i = numpy.random.randint(0, n_tets_sampled)
                while True:
                    new_tet = numpy.random.randint(0, len(Es[0]))
                    if new_tet in self.state:
                        pass
                    else:
                        break

                self.state[i] = new_tet

        def energy(self):
            U_bar = U[self.state, :]
            E_bars = Es[:, self.state]

            if True:
                sol = numpy.linalg.lstsq(U_bar, E_bars.T)
                alphas = sol[0]
                completion_mse = 1.0/(Es.shape[0] * Es.shape[1]) * (numpy.trace(alphas.transpose() @ UTU @ alphas) - numpy.trace(alphas.transpose() @ UTE)) 
            else:
                sol = numpy.linalg.solve(U_bar.T @ U_bar, U_bar.T)
                E_stars = (U_sum.T @ sol) @ E_bars.T
                completion_mse = mean_squared_error(E_stars, Es_summed)
            if completion_mse < self.min_mse:
                self.min_mse = completion_mse
            training_log['best_energy'].append(self.min_mse) 

            return completion_mse 

    init_state = a 
    prob = Problem(init_state, min_mse)
    prob.Tmax = 1 
    prob.Tmin = 0.00001
    prob.steps = 10000 
    prob.updates = prob.steps / 100 
    print("\nDetermining annealing schedule...")

    print("\nRunning simulated annealing...")
    a, b = prob.anneal()
    print("Done.")
    print("\nSelected tets:", a)

    print('\nmin mse (minus constant):', b)

    U_bar = U[a, :]
    E_bars = Es[:, a]
    sol = numpy.linalg.lstsq(U_bar, E_bars.T)
    completed_energies = (U @ sol[0]).T
    completion_mse = mean_squared_error(completed_energies, Es)
    print('Final reduced energy mse:', completion_mse, 'vs pca minimum of:', pca_train_mse)
    print()

    training_log["final_tets"] = a.tolist()
    training_log["final_mse"] = completion_mse
    training_log["pca_mse"] = pca_train_mse
    training_log["config"] = energy_model_config
    training_log["total_time"] = time.time() - start

    if do_save:
        my_utils.save_numpy_mat_to_dmat(index_output_path, numpy.ascontiguousarray(numpy.array(a,dtype=numpy.int32)))
        with open(os.path.join(model_root, 'energy_training_history.json'), 'w') as f:
            json.dump(training_log, f, indent=2)

def compute_relative_percent_error(pred, actual):
    return 100.0 * numpy.sum(numpy.abs(actual - pred)) / numpy.sum(numpy.abs(actual))

def sparse_learn_weights_l1(model_root, use_reencoded=False, use_extra_samples=False):
    V, T, F = my_utils.read_MESH_to_numpy(os.path.join(model_root, 'tets.mesh'))
    print("Computing volumes...")
    volumes = my_utils.compute_element_volumes(V, T)
    max_vol = numpy.max(volumes)
    volumes /= max_vol

    displacements, encoded_displacements, energies = load_displacements_and_energy(model_root, use_reencoded, use_extra_samples)

    energies = energies / numpy.apply_along_axis(numpy.linalg.norm, 1, energies)[:, None]

    flatten_data, unflatten_data = my_utils.get_flattners(energies)
    use_binary = False
    if use_binary:
        ae_decode, hist = binary_discrete_nn_analysis(
                                        encoded_displacements, flatten_data(energies),
                                        [], [],
                                        activation='elu',
                                        epochs=70,
                                        batch_size=512,
                                        layers=[200,50], 
                                        do_fine_tuning=False,
                                        model_root=model_root,
                                        autoencoder_config=None,
                                        energy_model_config={'enabled':False},
                                    )
    else:
        ae_decode, hist = discrete_nn_analysis(
                                        encoded_displacements, flatten_data(energies),
                                        [], [],
                                        volumes,
                                        activation='elu',
                                        epochs=400,
                                        batch_size=256,
                                        layers=[50,50,3], 
                                        do_fine_tuning=False,
                                        model_root=model_root,
                                        autoencoder_config=None,
                                        energy_model_config={'enabled':False},
                                    )

def learn_direct_energy(model_root, use_reencoded=False, use_extra_samples=False):
    displacements, encoded_displacements, energies = load_displacements_and_energy(model_root, use_reencoded, use_extra_samples)

    energies = numpy.sum(energies, axis=1, keepdims=True)

    ae_decode, hist = fit_nn_direct_energy(
                            encoded_displacements, energies,
                            [], [],
                            activation='elu',
                            epochs=200,
                            batch_size=512,
                            layers=[200,500], 
                            do_fine_tuning=False,
                            model_root=model_root,
                            autoencoder_config=None,
                            energy_model_config={'enabled':False},
                        )    

def pca_analysis(data, n_components, plot_explained_variance=False):
    print("Doing PCA with", n_components, "components...")
    pca = PCA(n_components=n_components)

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    flattened_data = flatten_data(data)
    pca.fit(flattened_data)

    if plot_explained_variance:
        explained_var = pca.explained_variance_ratio_
        plt.xlabel('Component')
        plt.ylabel('Explained Variance')
        plt.plot(list(range(1, n_components + 1)), explained_var)
        plt.show(block=False)

        print("Total explained variance =", sum(explained_var))

    mse = mean_squared_error(flattened_data, pca.inverse_transform(pca.transform(flattened_data)))

    def encode(decoded_data):
        X = flatten_data(decoded_data) - pca.mean_
        return numpy.dot(X, pca.components_.T)

    def decode(encoded_data):
        X_original = numpy.dot(encoded_data, pca.components_)
        X_original = X_original + pca.mean_
        return unflatten_data(X_original)

    return pca, encode, decode, pca.explained_variance_ratio_, mse

def discrete_nn_analysis(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    volumes,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): 

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    reduced_space_samples = flatten_data(reduced_space_samples)
    energy_samples = flatten_data(energy_samples)
    reduced_space_validation_samples = flatten_data(reduced_space_samples[:10] if reduced_space_validation_samples is None else reduced_space_validation_samples)
    energy_validation_samples = flatten_data(energy_samples[:10] if energy_validation_samples is None else energy_validation_samples)

    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    def lp_reg(p, lam):
        def reg(vec):
            return lam * K.sum(K.pow(K.abs(vec), p))
        return reg

    def l1_volume_reg(lam):
        vol = K.variable(volumes)
        def reg(vec):
            return lam * K.sum(K.abs(vec) * vol)

        return reg

    output = Dense(len(energy_samples[0]), activation='relu', name="output_layer" + str(i),
                    activity_regularizer=l1_volume_reg(0.001)
                    )(output)

    model = Model(input, output)

    def energy_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred * y_true, axis=-1) - K.sum(y_true, axis=-1))) 

    def energy_loss_numpy(y_true, y_pred):
        return numpy.mean(numpy.square(numpy.sum(y_pred * y_true, axis=-1) - numpy.sum(y_true, axis=-1)))

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss=energy_loss
    )
    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = energy_loss_numpy(actual, pred)
            print("Actual energy: ", (actual).sum(axis=1))
            print("Predicted energy: ", numpy.sum(pred * actual, axis=-1))
            nonzeros = [p[numpy.nonzero(p)] for p in pred]
            print("len(nonzero):", [len(nz) for nz in nonzeros])
            self.history.setdefault('mse', []).append(mse)
            print()
            print("Mean squared error: ", mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "l1_discrete_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist

def binary_discrete_nn_analysis(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): 

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda, multiply, ActivityRegularization
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    reduced_space_samples = flatten_data(reduced_space_samples)
    energy_samples = flatten_data(energy_samples)
    reduced_space_validation_samples = flatten_data(reduced_space_samples[:10] if reduced_space_validation_samples is None else reduced_space_validation_samples)
    energy_validation_samples = flatten_data(energy_samples[:10] if energy_validation_samples is None else energy_validation_samples)

    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    fixed_input = Input(tensor=K.variable([[1.0]] ))
    weights = Dense(len(energy_samples[0]), activation=None, use_bias=False)(fixed_input)
    output = Dense(len(energy_samples[0]), activation='hard_sigmoid', name="output_layer" + str(i),
                    )(output)

    output = multiply([output, weights])
    output = ActivityRegularization(l1=0.001, l2=0.0)(output)

    model = Model([input, fixed_input], output)

    def energy_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred * y_true, axis=-1) - K.sum(y_true, axis=-1))) 

    def energy_loss_numpy(y_true, y_pred):
        return numpy.mean(numpy.square(numpy.sum(y_pred * y_true, axis=-1) - numpy.sum(y_true, axis=-1)))

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss=energy_loss
    )
    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = energy_loss_numpy(actual, pred)
            print("Actual energy: ", (actual).sum(axis=1))
            print("Predicted energy: ", numpy.sum(pred * actual, axis=-1))
            nonzeros = [p[numpy.nonzero(p)] for p in pred]
            print("len(nonzero):", [len(nz) for nz in nonzeros])
            self.history.setdefault('mse', []).append(mse)
            print()
            print("Mean squared error: ", mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "l1_discrete_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist

def fit_nn_direct_energy(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): 

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    reduced_space_samples = flatten_data(reduced_space_samples)

    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    output = Dense(len(energy_samples[0]), activation='relu', name="output_layer" + str(i),
                    )(output)

    model = Model(input, output)

    optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_percentage_error',
    )
    def mean_absolute_percentage_error(y_true, y_pred): 

        return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = mean_absolute_percentage_error(actual, pred)
            print("Actual energy: ", actual)
            print("Predicted energy: ", pred)
            self.history.setdefault('mse', []).append(mse)
            print()
            print("Percentage error: ", mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "direct_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist

if __name__ == "__main__":
    main()
EOF
