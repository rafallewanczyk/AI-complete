
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

import os
import sys
import argparse

import pandas as pd
import math
import h5py
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme, ShuffledExampleScheme
from fuel.datasets.hdf5 import H5PYDataset

def net1(weights_path=None): 
    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(function=(lambda image: tf.image.resize_images(image, (48,48))), input_shape=(32,32,3)))
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(48,48,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv01'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv02'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv03'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv04'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv05'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv06'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv07'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv08'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv09'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv10'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv11'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv12'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net2(load_weights=False, load_local_weights=True): 
    if load_weights:
        if load_local_weights:
            model_vgg16_conv = keras.applications.vgg16.VGG16(weights=None, include_top=False) 
            model_vgg16_conv.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        else:
             model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False) 
    else:
        model_vgg16_conv = keras.applications.vgg16.VGG16(weights=None, include_top=False)
    input = keras.layers.Input(shape=(32,32,3),name='image_input')
    reshape_layer = keras.layers.Lambda(function=(lambda image: tf.image.resize_images(image, (48,48))), input_shape=(32,32,3))(input)
    output_vgg16_conv = model_vgg16_conv(reshape_layer)
    x = keras.layers.Flatten(name='flatten')(output_vgg16_conv)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(11, activation='softmax', name='predictions')(x)
    model = keras.models.Model(inputs=input, outputs=x)
    print model.summary()
    return model

def net3(weights_path=None): 
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net4(weights_path=None): 
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net5(weights_path=None): 
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv5'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv6'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv7'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net6(weights_path=None): 
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv01'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv02'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv03'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv04'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv05'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv06'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv07'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv08'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv09'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv10'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv11'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv12'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def dataset_generator(dataset, batch_size=500):
    while 1:
        trainstream = DataStream(dataset, iteration_scheme=ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size))
        for data in trainstream.get_epoch_iterator():
            images, labels = data
            m = images.mean(axis=(1,2,3), keepdims=True)
            images = (images - m)
            images = np.transpose(images, (0,2,3,1))
            labels = keras.utils.to_categorical(labels, num_classes=11)
            yield(images, labels)
        trainstream.close()

def dataset_generator1(dataset, handle, batch_size=500):
    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=True,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=0.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

    handle = dataset.open()
    data = dataset.get_data(handle, request = range(dataset.num_examples))    

    stream = datagen.flow(data[0], data[1], batch_size=batch_size)
    dataset.close(handle)
    return stream

def step_decay(epoch): 
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train(model=None):
    if model is not None:
        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), sources=('features', 'targets'))
        trainstream = DataStream(trainset, iteration_scheme=SequentialScheme(examples=trainset.num_examples, batch_size=500))
        for data in trainstream.get_epoch_iterator():
            images, labels = data
            m = images.mean(axis=(2,3), keepdims=True)
            s = images.std(axis=(2,3), keepdims=True)
            images = (images - m)/s
            images = np.transpose(images, (0,2,3,1))
            labels = keras.utils.to_categorical(labels)
            model.train_on_batch(x=images, y=labels)
        trainstream.close()

def train1(model=None):
    if model is not None:
        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), sources=('features', 'targets'))
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        batch_size = 500
        epochs_to_wait_for_improve = 1
        csv_logger = keras.callbacks.CSVLogger('traininglog.csv')
        check_point = keras.callbacks.ModelCheckpoint("model3epochweights.h5", monitor='val_loss', 
                                                    verbose=0, save_best_only=False, 
                                                    save_weights_only=True, mode='auto', period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=15, verbose=2,
                                        callbacks=[csv_logger, check_point, early_stopping],
                                        validation_data=dataset_generator(testset, batch_size),
                                        validation_steps=np.ceil(testset.num_examples/batch_size))
        return history

def train2(model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv"):
    if model is not None:
        dataset_size = 73257
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), 
                                sources=('features', 'targets'), subset=train_idx)
        validationset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), 
                                sources=('features', 'targets'), subset=validation_idx)
        batch_size = 500
        epochs_to_wait_for_improve = 15
        csv_logger = keras.callbacks.CSVLogger(log_save)
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=True, 
                                                        save_weights_only=True, mode='auto', period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=num_epochs, verbose=2,
                                        callbacks=[csv_logger, check_point, early_stopping],
                                        validation_data=dataset_generator(validationset, batch_size),
                                        validation_steps=np.ceil(validationset.num_examples/batch_size))
        save_model(model, weights, model_save)
        return history

def load_weights_to_model(model=None, weights_path=""):
    if weights_path and (model is not None):
        model.load_weights(weights_path)
        print "weights loaded from {}".format(weights_path)
    else:
        print "weights not loaded"

def train_no_aug(dataset_used, model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv", \
            reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,
            early_stopping_patience=1, continue_training=False): 
    if model is not None:
        dataset_size = H5PYDataset(dataset_used, which_sets=('train','train_neg')).num_examples
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainset = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('features', 'targets'), subset=train_idx)
        validationset = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('features', 'targets'), subset=validation_idx)
        batch_size = 500
        csv_logger = keras.callbacks.CSVLogger(log_save, append=continue_training)
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=False, 
                                                        save_weights_only=True, mode='auto', period=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                              patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, \
                                        write_grads=False, write_images=False)
        callback_list = [reduce_lr, csv_logger, check_point, early_stopping, tb]

        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=num_epochs, verbose=1,
                                        callbacks=callback_list,
                                        validation_data=dataset_generator(validationset, batch_size),
                                        validation_steps=np.ceil(validationset.num_examples/batch_size))
        save_model(model, weights, model_save)
        return history

def train_aug(dataset_used, model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv", \
            reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,
            early_stopping_patience=1, continue_training=False): 
    if model is not None:
        traingen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        valgen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        dataset_size = H5PYDataset(dataset_used, which_sets=('train','train_neg')).num_examples
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainsetX = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('features', ), subset=train_idx, load_in_memory=True)
        trainsetY = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('targets',), subset=train_idx, load_in_memory=True)
        validationsetX = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('features',), subset=validation_idx, load_in_memory=True)
        validationsetY = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('targets',), subset=validation_idx, load_in_memory=True)

        trainsetX, = trainsetX.data_sources
        validationsetX, = validationsetX.data_sources
        trainsetY, = trainsetY.data_sources
        validationsetY, = validationsetY.data_sources
        trainsetX = trainsetX - trainsetX.mean(axis=(1,2,3), keepdims=True)
        validationsetX = validationsetX - validationsetX.mean(axis=(1,2,3), keepdims=True)

        print trainsetX.shape
        print validationsetX.shape
        trainsetX = np.transpose(trainsetX, (0,2,3,1))
        validationsetX = np.transpose(validationsetX, (0,2,3,1))

        trainsetY = keras.utils.to_categorical(trainsetY, num_classes=11)
        validationsetY = keras.utils.to_categorical(validationsetY, num_classes=11)

        batch_size = 500
        csv_logger = keras.callbacks.CSVLogger(log_save, append=continue_training)
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=False, 
                                                        save_weights_only=True, mode='auto', period=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                              patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, \
                                        write_grads=False, write_images=False)
        callback_list = [reduce_lr, csv_logger, check_point, early_stopping, tb]

        history = model.fit_generator(traingen.flow(trainsetX, trainsetY, batch_size),
                                        steps_per_epoch=np.ceil(trainsetX.shape[0]/batch_size), 
                                        epochs=num_epochs, verbose=1,
                                        callbacks=callback_list,
                                        validation_data=valgen.flow(validationsetX, validationsetY, batch_size),
                                        validation_steps=np.ceil(validationsetX.shape[0]/batch_size))
        save_model(model, weights, model_save)
        return history

def test(model=None):
    if model is not None:
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        teststream = DataStream(testset, iteration_scheme=SequentialScheme(examples=testset.num_examples, batch_size=500))
        for data in teststream.get_epoch_iterator():
            images, labels = data
            images = np.swapaxes(images, axis1=1, axis2=3)
            labels = keras.utils.to_categorical(labels)
            loss, accuracy = model.test_on_batch(x=images, y=labels)
            accuracies.append(accuracy)
        trainstream.close()
        return losses

def test1(model=None):
    if model is not None:
        batch_size = 500
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        loss, accuracy = model.evaluate_generator(dataset_generator(testset, batch_size), 
                                                    steps=np.ceil(testset.num_examples/batch_size), 
                                                    max_queue_size=10, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy

def test_no_aug(dataset_used, model=None, testset=('test', 'test_neg',)): 
    if model is not None:
        batch_size = 500
        testset = H5PYDataset(dataset_used, which_sets=testset, sources=('features', 'targets')) 
        loss, accuracy = model.evaluate_generator(dataset_generator(testset, batch_size), 
                                                    steps=np.ceil(testset.num_examples/batch_size), 
                                                    max_queue_size=11, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy

def test_aug(dataset_used, model=None, testset=('test', 'test_neg',)): 
    if model is not None:

        testgen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        batch_size = 500
        testsetX = H5PYDataset(dataset_used, which_sets=testset, 
                                sources=('features', ), load_in_memory=True)
        testsetY = H5PYDataset(dataset_used, which_sets=testset, 
                                sources=('targets',), load_in_memory=True)
        testsetX, = testsetX.data_sources
        testsetY, = testsetY.data_sources
        testsetX = testsetX - testsetX.mean(axis=(1,2,3), keepdims=True)

        testsetX = np.transpose(testsetX, (0,2,3,1))
        testsetY = keras.utils.to_categorical(testsetY, num_classes=11)
        loss, accuracy = model.evaluate_generator(testgen.flow(testsetX, testsetY, batch_size), 
                                                    steps=np.ceil(testsetX.shape[0]/batch_size), 
                                                    max_queue_size=11, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy

def save_model(model, weights_path="", model_path=""):
    if model_path:
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
    if weights_path:
        model.save_weights(weights_path)

def train_model(dataset_used="new_more_neg.hdf5", model_num=1, attempt_num=None, comment="", weights_path="", num_epochs=100, \
                train_num=3, test_num=2, lr=1e-2, decay=1e-6, momentum=0.5, nesterov=True,\
                reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,\
                early_stopping_patience=50, continue_training=False):
    epoch_weights = "model{}epochweights{}_{}.h5".format(model_num, attempt_num, comment)
    weights = "model{}weights{}_{}.h5".format(model_num, attempt_num, comment)
    log_save = "model{}traininglog{}_{}.csv".format(model_num, attempt_num, comment)
    model_save = "model{}.json".format(model_num)

    tf.set_random_seed(0) 

    exec("model = net{}()".format(model_num))
    if weights_path:
        load_weights_to_model(model, weights_path)

    sgd = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print "learning rate is: {}".format(lr)
    exec("history = train{}(dataset_used=dataset_used, model=model, num_epochs=num_epochs, epoch_weights=epoch_weights, \
            weights=weights, model_save=model_save, log_save=log_save, \
            reduce_lr_patience=reduce_lr_patience, reduce_lr_min_lr=reduce_lr_min_lr, reduce_lr_factor=reduce_lr_factor,\
            early_stopping_patience=early_stopping_patience, continue_training=continue_training)".format(train_num))

    save_model(model, weights_path=weights, model_path=model_save)
    print "training completed, model saved to {}, weights saved to {}".format(model_save, weights)

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test', 'test_neg',))".format(test_num))
    print "test loss:", loss
    print "test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test',))".format(test_num))
    print "pos test loss:", loss
    print "pos test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test_neg',))".format(test_num))
    print "neg test loss:", loss
    print "neg test accuracy:", accuracy

def test_model(dataset_used, model_file="", model_num=1, weights="", test_num=2):
    exec("model = net{}()".format(model_num))

    model.load_weights(weights)
    sgd = keras.optimizers.SGD(lr=1e-1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    exec("loss, accuracy = test{}(dataset_used, model, testset=('test', 'test_neg',))".format(test_num))
    print "test loss:", loss
    print "test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test',))".format(test_num))
    print "pos test loss:", loss
    print "pos test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test_neg',))".format(test_num))
    print "neg test loss:", loss
    print "neg test accuracy:", accuracy

if __name__ == "__main__":
    task = "train"

    if task == "train":
        dataset_used = "new_more_neg_noresize.hdf5"
        comment = "40kneg_noresize_aug20_minus_mean"
        print "Note: {}".format(comment)
        train_model(dataset_used = dataset_used, model_num=5, attempt_num=1, comment=comment, \
                    weights_path="", num_epochs=300, \
                    train_num="_aug", test_num="_aug", lr=1e-2, decay=0, momentum=0.3, nesterov=True,\
                    reduce_lr_patience=5, reduce_lr_min_lr=1e-10, reduce_lr_factor=0.5,\
                    early_stopping_patience=20, continue_training=False)
        print "Note: {}".format(comment)
        print "data used: {}".format(dataset_used)

    if task == "test":

        dataset_used = "new_more_neg_noresize_80kneg.hdf5"
        test_model(dataset_used, model_file="model1.json", model_num=6, weights="model6weights1_80kneg_noresize_aug20_minus_mean.h5", test_num="_aug") 


EOF
