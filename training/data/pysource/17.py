from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from keras import backend as K
from keras.utils import plot_model
from keras.losses import mse, binary_crossentropy
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Lambda, Input, Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers
import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
from keras.utils import np_utils
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D

def dense_attension(inputs,num_units):
    dense1 = tf.keras.layers.Dense(num_units, activation="relu")(inputs)

    dense2 = tf.keras.layers.Dense(num_units, activation="relu")(inputs)
    dense2 = tf.keras.layers.Dense(num_units, activation="relu")(dense2)
    sigmoid = tf.keras.layers.Activation("sigmoid")(dense2)

    mul = tf.keras.layers.Multiply()([dense1,sigmoid])

    add = tf.keras.layers.Add()([inputs,mul])

    return add

def preprocess(df):
    x = df[["cp_wma"]]

    fit = preprocessing.MinMaxScaler().fit(x)
    x = fit.transform(x)

    x = np.append(x,0)

    gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        x, x, 50, batch_size=1)
    x = []
    y = []
    for i in range(len(gen)):
      xx, yy = gen[i]
      xx, yy = xx.tolist(), yy.tolist()
      x.extend(xx)
      y.extend(yy)

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        x, x, 1, batch_size=3000)
    x = []
    y = []
    for i in range(len(gen)):
      xx, yy = gen[i]
      xx, yy = xx.tolist(), yy.tolist()
      x.extend(xx)
      y.extend(yy)

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    x = x.reshape((x.shape[0], 50, 1))

def model_1():

    def mlti_res_block(inputs,filter_size1,filter_size2,filter_size3,filter_size4):
        cnn1 = tf.keras.layers.Conv1D(filter_size1,3,padding = 'causal',activation="relu")(inputs)
        cnn2 = tf.keras.layers.Conv1D(filter_size2,3,padding = 'causal',activation="relu")(cnn1)
        cnn3 = tf.keras.layers.Conv1D(filter_size3,3,padding = 'causal',activation="relu")(cnn2)

        cnn = tf.keras.layers.Conv1D(filter_size4,1,padding = 'causal',activation="relu")(inputs)

        concat = tf.keras.layers.Concatenate()([cnn1,cnn2,cnn3])
        mul = tf.keras.layers.Multiply()([concat,cnn])
        add = tf.keras.layers.Concatenate()([inputs,mul])

        return add
    inputs = tf.keras.layers.Input((6,1))

    reshape = tf.keras.layers.Permute((2, 1))(inputs)

    res_block = mlti_res_block(reshape,8,17,26,51)

    res_block = mlti_res_block(res_block,17,35,53,105)

    res_block = mlti_res_block(res_block,31,72,106,209)

    gru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=0.3, recurrent_dropout=0.3, implementation=1, return_sequences=True,
                                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00)))(res_block)
    gru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=0.3, recurrent_dropout=0.3, implementation=1, return_sequences=True,
                                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=0.00)))(gru)

    cnn1 = tf.keras.layers.Conv1D(31,3,padding = 'causal',activation="relu")(res_block)
    cnn2 = tf.keras.layers.Conv1D(72,3,padding = 'causal',activation="relu")(cnn1)
    cnn3 = tf.keras.layers.Conv1D(106,3,padding = 'causal',activation="relu")(cnn2)

    cnn = tf.keras.layers.Conv1D(209,1,padding = 'causal',activation="relu")(res_block)

    concat = tf.keras.layers.Concatenate()([cnn1,cnn2,cnn3])
    mul = tf.keras.layers.Multiply()([concat,cnn])
    add = tf.keras.layers.Concatenate()([res_block,mul])

    flatten = tf.keras.layers.Flatten()(add)

    outputs = tf.keras.layers.Dense(1)(flatten)

    model = tf.keras.Model(inputs,outputs)

    model.summary()

def multi_res_u_net():
    def mlti_res_block(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
        cnn1 = Conv1D(filter_size1, 3, padding='causal',
                  activation="relu")(inputs)
        cnn2 = Conv2D(filter_size2, 3, padding='causal',
                      activation="relu")(cnn1)
        cnn3 = Conv2D(filter_size3, 3, padding='causal',
                      activation="relu")(cnn2)

        cnn = Conv2D(filter_size4, 1, padding='causal',
                     activation="relu")(inputs)

        concat = layers.Concatenate()([cnn1, cnn2, cnn3])
        add = layers.Add()([concat, cnn])

        return add

    def res_path(inputs, filter_size, path_number):
        def block(x, fl):
            cnn1 = Conv2D(filter_size, 3, padding='causal',
                          activation="relu")(inputs)
            cnn2 = Conv2D(filter_size, 1, padding='causal',
                          activation="relu")(inputs)

            add = layers.Add()([cnn1, cnn2])

            return add

        cnn = block(inputs, filter_size)
        if path_number <= 3:
            cnn = block(cnn, filter_size)
            if path_number <= 2:
                cnn = block(cnn, filter_size)
                if path_number <= 1:
                    cnn = block(cnn, filter_size)

        return cnn

    def multi_res_u_net(pretrained_weights=None, input_size=(256, 256, 1), lr=0.001):
        inputs = layers.Input(input_size)

        res_block1 = mlti_res_block(inputs, 8, 17, 26, 51)

        res_block2 = mlti_res_block(res_block1, 17, 35, 53, 105)

        res_block3 = mlti_res_block(res_block2, 31, 72, 106, 209)

        res_block4 = mlti_res_block(res_block3, 71, 142, 213, 426)

        res_block5 = mlti_res_block(res_block4, 142, 284, 427, 853)

        res_path4 = res_path(res_block4, 256, 4)
        concat = layers.Concatenate()([res_block5, res_path4])

        res_block6 = mlti_res_block(concat, 71, 142, 213, 426)

        res_path3 = res_path(res_block3, 128, 3)
        concat = layers.Concatenate()([res_block6, res_path3])

        res_block7 = mlti_res_block(concat, 31, 72, 106, 212)

        res_path2 = res_path(res_block2, 64, 2)
        concat = layers.Concatenate()([res_block7, res_path2])

        res_block8 = mlti_res_block(concat, 17, 35, 53, 105)

        res_path1 = res_path(res_block1, 32, 1)
        concat = layers.Concatenate()([res_block8, res_path1])

        res_block9 = mlti_res_block(concat, 8, 17, 26, 51)

        flatten = layers.Flatten()(res_block9)
        outputs = layers.Dense(1)(flatten)
        model = tf.keras.Modedl(inputs, outputs)
        modle.compile(tf.keras.optimizer.Adam(lr), loss='mse', metrics=['mae'])

        return model
    model = multi_res_u_net()
    return model

def multi_res_u_net_v2(pretrained_weights=None, input_size=(10,1), lr=0.001):
    inputs = layers.Input(input_size)
    resshape = layers.Permute((2,1))(inputs)
    res_block = mlti_res_block(resshape, 8, 17, 26, 51)
    res_path = res_path(res_block, 32, 1)
    c = layers.Concatenate()([res_block,res_path,resshape])
    res_block = mlti_res_block(c, 17, 35, 53, 105)
    res_path = res_path(res_block, 64, 2)
    c = layers.Concatenate()([res_block,res_path,resshape])

    res_block = mlti_res_block(c, 31, 72, 106, 209)
    res_path = res_path(res_block, 128, 3)
    c = layers.Concatenate()([res_block, res_path, resshape])

    res_block = mlti_res_block(c, 71, 142, 213, 426)
    res_path = res_path(res_block, 256, 4)
    c = layers.Concatenate()([res_block,res_path,resshape])

    res_block = mlti_res_block(c, 142, 284, 427, 853)
    res_path = res_path(res_block, 256, 4)
    c = layers.Concatenate()([res_block,res_path,resshape])

    flatten = layers.Flatten()(c)
    dense = layers.Dense(128,activation="relu")(flatten)
    outputs = layers.Dense(1)(dense)
    model = tf.keras.Model(inputs, outputs)

def mlti_res_block(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
    cnn1 = Conv1D(filter_size1, 3, padding='causal',
                  activation="relu")(inputs)
    cnn2 = Conv1D(filter_size2, 3, padding='causal',
                  activation="relu")(cnn1)
    cnn3 = Conv1D(filter_size3, 3, padding='causal',
                  activation="relu")(cnn2)
    cnn = Conv1D(filter_size4, 1, padding='causal',
                 activation="relu")(inputs)
    concat = layers.Concatenate()([cnn1, cnn2, cnn3])
    add = layers.Add()([concat, cnn])
    return add
def res_path1(inputs, filter_size, path_number):
    def block(x, fl):
        cnn1 = Conv1D(filter_size, 3, padding='causal',
                      activation="relu")(inputs)
        cnn2 = Conv1D(filter_size, 1, padding='causal',
                      activation="relu")(inputs)
        add = layers.Add()([cnn1, cnn2])
        return add
    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn, filter_size)
        if path_number <= 2:
            cnn = block(cnn, filter_size)
            if path_number <= 1:
                cnn = block(cnn, filter_size)
    return cnn
def multi_res_u_net(pretrained_weights=None, input_size=(30,3), lr=0.001):
    inputs = layers.Input(input_size)
    resshape = layers.Permute((2,1))(inputs)
    resshape = tf.keras.layers.MaxPool1D(3)(resshape)
    res_block = mlti_res_block(resshape, 8, 17, 26, 51)
    res_path = res_path1(res_block, 32, 1)
    c = layers.Concatenate()([res_block,resshape])
    res_block = mlti_res_block(c, 17, 35, 53, 105)
    res_path = res_path1(res_block, 64, 2)
    c = layers.Concatenate()([res_block,resshape])

    res_block = mlti_res_block(c, 31, 72, 106, 209)
    res_path = res_path1(res_block, 128, 3)
    c = layers.Concatenate()([res_block, resshape])

    res_block = mlti_res_block(c, 71, 142, 213, 426)
    res_path = res_path1(res_block, 256, 4)
    c = layers.Concatenate()([res_block,resshape])

    res_block = mlti_res_block(c, 142, 284, 427, 853)
    res_path = res_path1(res_block, 256, 4)
    c = layers.Concatenate()([res_block,resshape,res_path])
    flatten = layers.Flatten()(c)
    outputs = layers.Dense(1)(flatten)
    model = tf.keras.Model(inputs, outputs)
    return model

inputs = layers.Input((30, 1))

lstm = layers.CuDNNLSTM(32, return_sequences=True)(inputs)
lstm1 = layers.CuDNNLSTM(32, return_sequences=True)(lstm)

concat = layers.Concatenate()([lstm, lstm1])

lstm2 = layers.CuDNNLSTM(64, return_sequences=True)(inputs)

add = layers.Add()([concat, lstm2])

cnn = Conv1D(12, 3, activation="relu", padding="causal")(add)
cnn = Conv1D(24, 3, activation="relu", padding="causal")(cnn)
cnn = Conv1D(36, 3, activation="relu", padding="causal")(cnn)

flatten = layers.Flatten()(cnn)
outputs = layers.Dense(1)(flatten)

model = tf.keras.Model(inputs, outputs)

def conv_net(inputs, rate):
  conv1 = Conv1D(17,2,padding="causal", dilation_rate=rate)(inputs)
  conv1 = layers.PReLU()(conv1)
  concat = layers.Concatenate()([inputs,conv1])
  conv2 = Conv1D(35,2,padding="causal",dilation_rate=rate)(concat)
  conv2 = layers.PReLU()(conv2)
  concat = layers.Concatenate()([concat,conv2])
  conv3 = Conv1D(53,2,padding="causal",dilation_rate=rate)(concat)
  conv3 = layers.PReLU()(conv3)
  concat = layers.Concatenate()([concat,conv3])
  a = inputs.shape[2]
  a = np.int(a)
  a += 105

  conv4 = Conv1D(a,1,padding="causal",activation="relu",dilation_rate=rate)(inputs)

  add = layers.Add()([concat,conv4])
  return add

conv1 = conv_net(inputs,1)
conv2 = conv_net(inputs,1)
conv3 = conv_net(inputs,8)
conv4 = conv_net(inputs,16)

add = layers.Add()([conv1,conv2])

conv = Conv1D(64,2,padding="causal")(add)
conv = tf.keras.layers.PReLU()(conv)
concat = layers.Concatenate()([inputs,add,conv])

conv = Conv1D(64,2,padding="causal")(concat)
conv = tf.keras.layers.PReLU()(conv)
concat = layers.Concatenate()([concat,conv])

conv = Conv1D(64,2,padding="causal")(concat)
conv = tf.keras.layers.PReLU()(conv)
concat = layers.Concatenate()([concat,conv])

        conv1 = Conv1D(8,ks,padding="causal",activation="relu")(inputs)
        concat = layers.Concatenate()([inputs,conv1])
        conv2 = Conv1D(17,ks,padding="causal",activation="relu")(concat)
        concat = layers.Concatenate()([concat,conv2])
        conv3 = Conv1D(26,ks,padding="causal",activation="relu")(concat)
        concat = layers.Concatenate()([concat,conv3])

        a = inputs.shape[2]
        a = np.int(a)
        a += 51

        conv4 = Conv1D(a,1,padding="causal",activation="relu")(inputs)

        add = layers.Add()([concat,conv4])

        return concat 

def fine(inputs):
    cnn = Conv1D(18,3,padding="causal",activation="relu")(inputs)
    concat = layers.Concatenate()([inputs,cnn])
    cnn = Conv1D(18,3,padding="causal",activation="relu")(concat)
    concat = layers.Concatenate(concat,cnn)
    cnn = Conv1D(36,3,padding="causal",activation="relu")(concat)
    concat = layers.Concatenate()([concat,cnn])

    return concat

def midium(inputs):
    cnn = Conv1D(18,3,padding="causal",activation="relu")(inputs)
    concat = layers.Concatenate()([inputs,cnn])
    cnn = Conv1D(36,3,padding="causal",activation="relu")(concat)
    concat = layers.Concatenate()([concat,cnn])

    return concat
def coarse(inputs):
    cnn = Conv1D(36,3,padding="causal",activation="relu")(inputs)
    concat = layers.Concatenate()([inputs,cnn])

    return concat

def fine(inputs):
    cnn = Conv1D(36, 4, padding="same")(inputs)
    cnn = layers.PReLU()(cnn)
    cnn = Conv1D(36, 4, padding="same")(cnn)
    cnn = layers.PReLU()(cnn)
    cnn = Conv1D(72, 4, padding="same")(cnn)
    cnn = layers.PReLU()(cnn)
    return cnn
def midium(inputs):
    cnn = Conv1D(36, 3, padding="same")(inputs)
    cnn = layers.PReLU()(cnn)
    cnn = Conv1D(72, 3, padding="same")(cnn)
    cnn = layers.PReLU()(cnn)
    return cnn

def coarse(inputs):
    cnn = Conv1D(72, 2, padding="same")(inputs)
    cnn = layers.PReLU()(cnn)
    concat = layers.Concatenate()([inputs, cnn])
    return cnn
inputs = layers.Input((num_nits, 1))
inputs_2 = layers.MaxPool1D()(inputs)
inputs_3 = layers.MaxPool1D(4)(inputs)
net1 = fine(inputs=inputs)
net2 = midium(inputs_2)
net3 = coarse(inputs_3)
net2 = layers.UpSampling1D()(net2)
net3 = layers.UpSampling1D(4)(net3)
concat = layers.Add()([net1, net2, net3])
concat = layers.Flatten()(concat)
dense = layers.Dense(128, activation="relu")(concat)
outputs = layers.Dense(s)(concat)
model = tf.keras.Model(inputs, outputs)

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D

def coarse_fine(inputs):
    def Net1(inputs):
        cnn = Conv1D(18,6,padding="causal",activation="relu")(inputs)
        cnn = Conv1D(18,6,padding="causal",activation="relu")(cnn)
        cnn = Conv1D(36,6,padding="causal",activation="relu")(cnn)
        cnn = Conv1D(36,6,padding="causal",activation="relu")(cnn)
        cnn = Conv1D(74,6,padding="causal",activation="relu")(cnn)
        cnn = Conv1D(74, 6, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(148, 6, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(148, 6, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(296, 6, padding="causal", activation="relu")(cnn)
        return cnn
    def Net2(inputs):
        pool = layers.MaxPool1D(2)(inputs)
        cnn = Conv1D(18,8,padding="causal",activation="relu")(pool)
        cnn = Conv1D(36,8,padding="causal",activation="relu")(cnn)    
        cnn = Conv1D(74, 8, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(148, 8, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(296, 6, padding="causal", activation="relu")(cnn)
        return cnn

    def Net3(inputs):
        pool = layers.MaxPool1D(4)(inputs)
        cnn = Conv1D(36,8,padding="causal",activation="relu")(pool)
        cnn = Conv1D(74, 8, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(148, 8, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(296, 8, padding="causal", activation="relu")(cnn)

        return cnn
    def Net4(inputs):
        pool = layers.MaxPool1D(8)(inputs)
        cnn = Conv1D(74, 6, padding="causal", activation="relu")(pool)
        cnn = Conv1D(148, 6, padding="causal", activation="relu")(cnn)
        cnn = Conv1D(296, 6, padding="causal", activation="relu")(cnn)

        return cnn

    def Net5(inputs):
        pool = layers.MaxPool1D(16)(inputs)
        cnn = Conv1D(148, 4, padding="causal", activation="relu")(pool)
        cnn = Conv1D(296, 4, padding="causal", activation="relu")(cnn)

        return cnn

    def Net6(inputs):
        pool = layers.MaxPool1D(32)(inputs)
        cnn = Conv1D(296, 2, padding="causal", activation="relu")(pool)

        return cnn

    net1 = Net1(inputs=inputs)
    net2 = Net2(inputs=inputs)
    net3 = Net3(inputs=inputs)
    net4 = Net4(inputs=inputs)
    net5 = Net5(inputs=inputs)
    net6 = Net6(inputs=inputs)

    net2 = layers.UpSampling1D(2)(net2)
    net3 = layers.UpSampling1D(4)(net3)
    net4 = layers.UpSampling1D(8)(net4)
    net5 = layers.UpSampling1D(16)(net5)
    net6 = layers.UpSampling1D(32)(net6)

    concat = layers.Add()([net1,net2,net3,net5,net6])
    concat = layers.Flatten()(concat)
    outputs = layers.Dense(1)(concat)

    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
    return model

inputs = layers.Input((num_nits,1))

model = coarse_fine(inputs)

def Model(inputs_shape=(None,None), outputs_shape=None):
    inputs = layers.Input(inputs_shape)

    cnn1 = layers.Conv1D(32,8,padding="causal",activation="relu")(inputs)
    cnn2 = layers.Conv1D(32,4,padding="causal",activation="relu")(inputs)
    cnn3 = layers.Conv1D(32,2,padding="causal",activation="relu")(inputs)
    cnn4 = layers.Conv1D(32,8,padding="causal",activation="relu")(inputs)

    add1 = layers.Add()([cnn1,cnn2,cnn3,cnn4])

    cnn1 = layers.Conv1D(48,8,padding="causal",activation="relu")(add1)
    cnn2 = layers.Conv1D(48,4,padding="causal",activation="relu")(add1)
    cnn3 = layers.Conv1D(48,2,padding="causal",activation="relu")(add1)
    cnn4 = layers.Conv1D(48,8,padding="causal",activation="relu")(add1)

    add2 = layers.Add()([cnn1,cnn2,cnn3,cnn4])

    cnn1 = layers.Conv1D(64,8,padding="causal",activation="relu")(add2)
    cnn2 = layers.Conv1D(64,4,padding="causal",activation="relu")(add2)
    cnn3 = layers.Conv1D(64,2,padding="causal",activation="relu")(add2)
    cnn4 = layers.Conv1D(64,8,padding="causal",activation="relu")(add2)

    add3 = layers.Add()([cnn1,cnn2,cnn3,cnn4])

    cnn1 = layers.Conv1D(64,8,padding="causal",activation="relu")(add3)
    cnn2 = layers.Conv1D(64,4,padding="causal",activation="relu")(add3)
    cnn3 = layers.Conv1D(64,2,padding="causal",activation="relu")(add3)
    cnn4 = layers.Conv1D(64,8,padding="causal",activation="relu")(add3)

    add3 = layers.Add()([cnn1,cnn2,cnn3,cnn4,add3])

    cnn1 = layers.Conv1D(48,8,padding="causal",activation="relu")(add3)
    cnn2 = layers.Conv1D(48,4,padding="causal",activation="relu")(add3)
    cnn3 = layers.Conv1D(48,2,padding="causal",activation="relu")(add3)
    cnn4 = layers.Conv1D(48,8,padding="causal",activation="relu")(add3)

    add2 = layers.Add()([cnn1,cnn2,cnn3,cnn4,add2])

    cnn1 = layers.Conv1D(32,8,padding="causal",activation="relu")(add2)
    cnn2 = layers.Conv1D(32,4,padding="causal",activation="relu")(add2)
    cnn3 = layers.Conv1D(32,2,padding="causal",activation="relu")(add2)
    cnn4 = layers.Conv1D(32,8,padding="causal",activation="relu")(add2)

    add1 = layers.Add()([cnn1,cnn2,cnn3,cnn4,add1])

    cnn = layers.Conv1D(32,2,padding="causal",activation="relu")(add1)

    flatten = layers.Flatten()(cnn)
    outputs = layers.Dense(1)(flatten)

    model = tf.keras.Model(inputs,outputs)

    return  model

inputs = layers.Input((num_nits,1))

cnn1 = layers.Conv1D(36,8,padding="causal",activation="relu")(inputs)

cnn2 = layers.Conv1D(36,4,padding="causal",activation="relu")(inputs)

cnn3 = layers.Conv1D(36,2,padding="causal",activation="relu")(inputs)

cnn4 = layers.Conv1D(109,1,padding="causal",activation="relu")(inputs)

add = layers.Add()([cnn1,cnn2,cnn3])
add = layers.Concatenate()([add,cnn4,inputs])

cnn1 = layers.Conv1D(36,8,padding="causal",activation="relu")(add)

cnn2 = layers.Conv1D(36,4,padding="causal",activation="relu")(add)

cnn3 = layers.Conv1D(36,2,padding="causal",activation="relu")(add)

cnn4 = layers.Conv1D(109,1,padding="causal",activation="relu")(add)

add = layers.Add()([cnn1,cnn2,cnn3])
add = layers.Concatenate()([add,cnn4])

cnn = Conv1D(36,2,padding="same",activation="relu")(add)
cnn1 = Conv1D(72,2,padding="causal",activation="relu")(add)

cnn = layers.Concatenate()([cnn,cnn1])

add = layers.Flatten()(cnn)
outputs = layers.Dense(1)(add)

model = tf.keras.Model(inputs,outputs)

inputs = layers.Input((num_nits,1))

cnn1 = layers.Conv1D(36,8,padding="causal",activation="relu")(inputs)

cnn2 = layers.Conv1D(36,4,padding="causal",activation="relu")(inputs)

cnn3 = layers.Conv1D(36,2,padding="causal",activation="relu")(inputs)

cnn4 = layers.Conv1D(109,1,padding="causal",activation="relu")(inputs)

add = layers.Add()([cnn1,cnn2,cnn3])

cnn = layers.Conv1D(36,2,padding="causal",activation="relu")(add)

add = layers.Concatenate()([cnn,cnn4,inputs])

cnn1 = layers.Conv1D(36,8,padding="causal",activation="relu")(add)

cnn2 = layers.Conv1D(36,4,padding="causal",activation="relu")(add)

cnn3 = layers.Conv1D(36,2,padding="causal",activation="relu")(add)

cnn4 = layers.Conv1D(109,1,padding="causal",activation="relu")(add)

add = layers.Add()([cnn1,cnn2,cnn3])

cnn = layers.Conv1D(36,2,padding="causal",activation="relu")(add)

add = layers.Concatenate()([cnn,cnn4])

cnn = Conv1D(36,2,padding="same",activation="relu")(add)
cnn1 = Conv1D(72,2,padding="causal",activation="relu")(add)

cnn = layers.Concatenate()([cnn,cnn1])

add = layers.Flatten()(cnn)
outputs = layers.Dense(1)(add)

model = tf.keras.Model(inputs,outputs)

inputs = layers.Input((num_nits,1))

cnn1 = layers.Conv1D(36,2,padding="causal",activation="relu")(inputs)

cnn2 = layers.Conv1D(36,8,padding="causal",activation="relu")(inputs)

cnn3 = layers.Conv1D(36,4,padding="causal",activation="relu")(inputs)

cnn4 = layers.Conv1D(108,1,padding="causal",activation="relu")(inputs)

add = layers.Add()([cnn1,cnn2,cnn3])

add = layers.Concatenate()([add,cnn4])

cnn = Conv1D(72,2,padding="causal",activation="relu")(add)
cnn1 = Conv1D(72,2,padding="causal",activation="relu", dilation_rate=2)(inputs)

add = layers.Add()([cnn,cnn1])

add = layers.Flatten()(add)
outputs = layers.Dense(1)(add)

model = tf.keras.Model(inputs,outputs)

inputs = layers.Input((num_nits,1))

cnn1 = layers.Conv1D(36,2,padding="causal",activation="relu")(inputs)

cnn2 = layers.Conv1D(36,8,padding="causal",activation="relu")(inputs)

cnn3 = layers.Conv1D(36,4,padding="causal",activation="relu")(inputs)

cnn4 = layers.Conv1D(108,1,padding="causal",activation="relu")(inputs)

add = layers.Add()([cnn1,cnn2,cnn3])

add1 = layers.Concatenate()([add,cnn4])

cnn1 = layers.Conv1D(36,8,padding="same",activation="relu", dilation_rate=4)(inputs)

cnn2 = layers.Conv1D(36,4,padding="same",activation="relu", dilation_rate=4)(inputs)

cnn3 = layers.Conv1D(36,2,padding="same",activation="relu", dilation_rate=4)(inputs)

cnn4 = layers.Conv1D(108,1,padding="same",activation="relu", dilation_rate=4)(inputs)

add = layers.Add()([cnn1,cnn2,cnn3])
add2 = layers.Concatenate()([add,cnn4])

add = layers.Concatenate()([add1,add2])

cnn = Conv1D(72,2,padding="causal",activation="relu")(add)
cnn1 = Conv1D(72,2,padding="causal",activation="relu", dilation_rate=4)(inputs)

add = layers.Add()([cnn,cnn1])

add = layers.Flatten()(add)
outputs = layers.Dense(1)(add)

model = tf.keras.Model(inputs,outputs)

def model():
    def block(inputs,type=1):
        cnn1 = layers.Conv1D(4,2,padding="causal",activation="relu")(inputs)
        cnn2 = layers.Conv1D(8,2,padding="causal",activation="relu")(inputs)
        cnn3 = layers.Conv1D(16,2,padding="causal",activation="relu")(inputs)
        cnn4 = layers.Conv1D(32,2,padding="causal",activation="relu")(inputs)
        cnn5 = layers.Conv1D(64,2,padding="causal",activation="relu")(inputs)

        concat =  layers.Concatenate()([cnn1,cnn2,cnn3,cnn4,cnn5])

        if type  == 1:
            concat = layers.ReLU()(concat)
        return concat

    inputs = layers.Input((num_nits,1))

    block1 = block(inputs)

    block2 = block(block1)
    block3 = block(block1)

    add = layers.Add()([block1,block2,block3])

    block4 = block(add)

    flatten = layers.Flatten()(block4)
    outputs = layers.Dense(1)(flatten)

    model = tf.keras.Model(inputs,outputs)
    tf.keras.utils.plot_model(model)

    return model


EOF
