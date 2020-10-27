

__author__ = "Elias Aoun Durand"
__email__ = "elias.aoundurand@gmail.com"

from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

from numba import jit
import cv2
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
import keras

import time
import random
import scipy
import math
import matplotlib.animation as animation
import tqdm

from keras.models import Model
from keras.layers import Activation, Dense, Input, Multiply
from keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.layers import Dot, Lambda, Concatenate, RepeatVector
from keras.utils import plot_model
from PIL import Image
from keras.constraints import max_norm, non_neg

matplotlib.rcParams.update({'font.size': 16})

L1 = 0.28
L2 = 0.28
L3 = 0.09
IMG_SIZE = 128
INPUT_ENCODER_SHAPE = (IMG_SIZE, IMG_SIZE, 2)
LATENT_DIM = 32
INPUT_DECODER_SHAPE = (1, 2 * LATENT_DIM,)

NB_POSTURE = 50
NB_COMMAND = 100
NB_DATA = NB_POSTURE*NB_COMMAND
BATCH_SIZE = 100
TEST_BUF = 1000

DIMS = (IMG_SIZE, IMG_SIZE,2)
N_TRAIN_BATCHES =int(NB_DATA/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

def randrange(n, vmin, vmax):
    return (vmax - vmin) * rand(n) + vmin

def control_robot(angles):
    phi1, phi2, theta1, psi1, psi2 = angles
    x = L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1)+ L3*cos(phi2+theta1+psi1)*cos(phi1+psi2)
    y =  L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1) + L3*cos(psi1+phi2+theta1)*sin(phi1+psi2)
    z = L1 * sin(phi2) + L2 * sin(phi2 +theta1) + L3*sin(psi1+phi2+theta1)  

    return np.array([x, y,z])

def control_robot_elbow(angles):
    phi1, phi2, theta1, psi1, psi2 = angles
    x = L1 * cos(phi1) * cos(phi2)
    y = L1 * cos(phi1) * sin(phi2)
    z = L1 * sin(phi1)
    return np.array([x,y,z])

def compute_trajectory(postures):
    tmp = []
    for i in range(len(postures)):
        tmp.append(control_robot(postures[i][0]))
    return np.array(tmp)

def compute_elbow_trajectory(postures):
    tmp = []
    for i in range(len(postures)):
        tmp.append(control_robot_elbow(postures[i][0]))
    return np.array(tmp)

def plot_arm(angles, time):
    phi1, phi2, theta1, psi1, psi2 = angles
    filename = 'images/%s.png' % time

    x = [0, 0, L1 * cos(phi1) * cos(phi2),
         L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1),
         L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1)+ L3*cos(phi2+theta1+psi1)*cos(phi1+psi2)]
    y = [0, 0, L1 * sin(phi1) * cos(phi2),
         L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1),
         L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1) + L3*cos(psi1+phi2+theta1)*sin(phi1+psi2)]
    z = [0, 0, L1 * sin(phi2),
         L1 * sin(phi2) + L2 * sin(phi2+theta1)  ,
         L1 * sin(phi2) + L2 * sin(phi2 +theta1) + L3*sin(psi1+phi2+theta1)]

    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.5, right=0.5)
    ax.set_ylim(bottom=-0.5, top=0.5)
    ax.set_zlim(bottom=-0.5, top=0.5)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
    close()

    return ax

def create_random_data(nb_posture, nb_command, typ='train'):

    posture = zeros((nb_posture, 5))
    posture[:, 0] = randrange(nb_posture, 0, pi)
    posture[:, 1] = randrange(nb_posture, -pi/2,  pi/2)
    posture[:, 2] = randrange(nb_posture, 0,  pi)
    posture[:, 3] = randrange(nb_posture, -pi/2, pi/2)
    posture[:, 4] = randrange(nb_posture, -pi/4,  pi/4)

    command = zeros((nb_command, 5))
    command[:, 0] = randrange(nb_command, -1, 1) * 0.1
    command[:, 1] = randrange(nb_command, -1, 1) * 0.1
    command[:, 2] = randrange(nb_command, -1, 1) * 0.1
    command[:, 3] = randrange(nb_command, -1, 1) * 0.1
    command[:, 4] = randrange(nb_command, -1, 1) * 0.1

    nb_data = nb_posture * nb_command

    train_data_x = zeros((nb_data, 1, 5))
    train_data_y = zeros((nb_data, 1, 5))
    train_data_h = zeros((nb_data, 1, 5))

    train_pos_x = zeros((nb_data, 1, 3))
    train_pos_y = zeros((nb_data, 1, 3))

    idx = 0
    for i in tqdm.tqdm(range(nb_posture), desc="train_data 1"):
        for j in range(nb_command):
            train_data_x[idx] = posture[i]
            train_data_y[idx] = posture[i] + command[j]
            train_data_h[idx] = command[j]

            tmp = control_robot(posture[i])
            ttmp = control_robot(posture[i] + command[j])

            train_pos_x[idx] = tmp
            train_pos_y[idx] = ttmp
            idx = idx + 1

    for i in tqdm.tqdm(range(nb_data), desc='figsave'):
        pos0, pos1, pos2, pos3, pos4 = train_data_x[i][0]
        dpos0, dpos1, dpos2, dpos3, dpos4 = train_data_y[i][0]

        before = typ + '/fig_before_%s' % i
        after = typ + '/fig_after_%s' % i
        plot_arm(train_data_x[i][0], before)
        plot_arm(train_data_y[i][0], after)

    return train_data_x, train_data_y, train_data_h, train_pos_x, train_pos_y

def sort_pictures(train_pos_x, train_pos_y, motion="up"):
    list_idx = sort_command(train_pos_x, train_pos_y, "up")

    for i in tqdm.tqdm(list_idx):
        before = 'images/' + typ + '/fig_before_%s.png' % i
        after = 'images/' + typ + '/fig_after_%s.png' % i

        tens_before = load_and_preprocess_image(before)
        tens_after = load_and_preprocess_image(after)

        noised_tens_before = noised_image(tens_before)
        noised_tens_after = noised_image(tens_after)

        t = tf.concat([noised_tens_before, noised_tens_after], -1)

        tf.reshape(t, [IMG_SIZE, IMG_SIZE, 2])
        tmp.append(t)

    return tf.stack(tmp)

def sort_command(train_pos_x, train_pos_y, motion):

    list_idx = []
    n = len(train_pos_x)

    if motion == "up":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][2] < train_pos_y[i][0][2]):
                list_idx.append(i)

    if motion == "down":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_y[i][0][2] > train_pos_y[i][0][2]):
                list_idx.append(i)

    if motion == "right":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][1] < train_pos_y[i][0][1]):
                list_idx.append(i)

    if motion == "left":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][1] > train_pos_y[i][0][1]):
                list_idx.append(i)

    if motion == "in":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][2][0] > train_pos_y[i][2][0]):
                list_idx.append(i)

    if motion == "out":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][2][0] < train_pos_y[i][2][0]):
                list_idx.append(i)

    return list_idx

def visual_direction(train_pos_x, train_pos_y):
    return train_pos_y - train_pos_x

def gaussian_kernel(size, mean, std):

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def load_and_process_images(nb_data, typ):
    tmp = []

    for i in tqdm.tqdm(range(nb_data)):
        before = 'images/' + typ + '/fig_before_%s.png' % i
        after = 'images/' + typ + '/fig_after_%s.png' % i

        tens_before = load_and_preprocess_image(before)
        tens_after = load_and_preprocess_image(after)

        noised_tens_before = noised_image(tens_before)
        noised_tens_after = noised_image(tens_after)

        t = tf.concat([noised_tens_before, noised_tens_after], -1)

        tf.reshape(t, [IMG_SIZE, IMG_SIZE, 2])
        tmp.append(t)

    return tf.stack(tmp)

def noised_image(tens):
    tens_shape = shape(tens)
    tmp = tf.convert_to_tensor(np.random.random(
        tens_shape), dtype='float32') * 0.1

    return tf.add(tmp, tens)

def preprocess_image(img):
    tmp = tf.image.decode_png(img, channels=1)
    tmp = tf.image.resize(tmp, [IMG_SIZE, IMG_SIZE])
    tmp /= 255.0  
    return tmp

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)

@tf.function
def compute_conv_loss(model, img, filter_index):
    output = model(img)
    loss = tf.keras.backend.mean(output[:, :, :, filter_index])
    return loss

def generate_conv_pattern(model, filter_index, nb_pass):
    input_img_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 1)), dtype='float32') * 2 + 1.
    tmp = tf.compat.v1.get_variable(
        'tmp', dtype=tf.float32, initializer=input_img_data)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tmp)
        loss = compute_conv_loss(model, tmp, filter_index)

    for i in range(nb_pass):
        grads = tape.gradient(loss, tmp, unconnected_gradients='zero')
        tmp.assign_add(grads)

    return tmp[0][:, :, :]

@tf.function
def compute_loss(model, img, filter_index):
    output = model(img)
    loss = (output[0][:, filter_index])
    return loss

def generate_pattern(model, filter_index, nb_pass):

    input_img_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 2)), dtype='float32') * 2 + 1.
    tmp = tf.compat.v1.get_variable(
        'tmp', dtype=tf.float32, initializer=input_img_data)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tmp)
        loss = compute_loss(model, tmp, filter_index)

    for i in range(nb_pass):
        grads = tape.gradient(loss, tmp, unconnected_gradients='zero')
        tmp.assign_add(grads)

    return tmp[0][:, :, :1]

def plot_and_compute_conv_filters(model, size=IMG_SIZE, margin=5, nb_pass=100000):
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))

    for i in tqdm.tqdm(range(8)):
        for j in tqdm.tqdm(range(8)):
            filter_img = generate_conv_pattern(model, i + (j * 8), nb_pass)
            horizontal_start = i * size + i + margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img

    return results

def plot_and_compute_last_filters(model, size=IMG_SIZE, margin=5, nb_pass=10000):
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))
    input_image_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 2)), dtype='float32') * 2 + 1.
    t = []
    j = 3
    for i in tqdm.tqdm(range(8)):
        for j in tqdm.tqdm(range(4), leave=False):
            filter_img = generate_pattern(model, i + (j * 8), nb_pass)
            horizontal_start = i * size + i + margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img
            t.append(filter_img)

    return t, results

def plot_and_save_visual_direction(train_position_before, train_position_after):
    visual = visual_direction(train_position_before, train_position_after)
    for i in tqdm.tqdm(range(len(train_position_before))):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(False)
        ax.axis("off")
        q = ax.quiver(train_position_before[i,0,0],
                  train_position_before[i,0,1],
                  train_position_before[i,0,2],
                  visual[i, 0, 0],
                  visual[i, 0, 1],
                  visual[i, 0, 2],
                  length = 0.1,
                  linewidth=20,
                  cmap='Reds')
        filename = 'images/visual_direction/%s.png' %i
        savefig(filename,  facecolor=fig.get_facecolor(), edgecolor='none')
        close()

def compute_latent_filters(model, list, iterator, nb_data):

    t = []
    color_position = []
    for i in tqdm.tqdm(range(nb_data)):
        tmp = iterator.get_next()
        tmp = tf.expand_dims(tmp, 0)

        j = check_color_position(list, i)
        t.append(model.predict(tmp))
        color_position.append(j)

    return t, color_position

def check_color_position(list, i):
    tmp = list[i]
    x, y, z = tmp[0][0], tmp[0][1], tmp[0][2]

    if (x > 0) and (y > 0) and (z > 0):
        j = 0
    elif (x > 0) and (y > 0) and (z < 0):
        j = 1
    elif (x > 0) and (y < 0) and (z > 0):
        j = 2
    elif (x > 0) and (y < 0) and (z < 0):
        j = 3
    elif (x < 0) and (y > 0) and (z > 0):
        j = 4
    elif (x < 0) and (y > 0) and (z < 0):
        j = 5
    elif (x < 0) and (y < 0) and (z > 0):
        j = 6
    elif (x < 0) and (y < 0) and (z < 0):
        j = 7
    else:
        j = 7

    return j

def build_dense_encoder(custom_shape=INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Dense(
        LATENT_DIM, activation='relu', name='latent_enc_fx1')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Dense(
        LATENT_DIM, activation='relu', name='latent_enc_fy1')(fy)
    fy = tf.keras.layers.Reshape((1, LATENT_DIM,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder

def build_dense_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fh)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fy1')(fy)

    y = tf.keras.layers.Dense(
        IMG_SIZE * IMG_SIZE, activation='relu', name='y_recon')(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder

def build_conv2D_encoder(custom_shape=INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_x_1')(x)
    fx = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fx)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1, 1),activation='relu', name='conv_x_2')(fx)
    fx = tf.keras.layers.Flatten()(fx)
    fx = tf.keras.layers.Dense(LATENT_DIM, name='latent_fx2')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_y_1')(y)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(2, 2), activation='relu', name='conv_y_2')(fy)
    fy = tf.keras.layers.Flatten()(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_fy2')(fy)
    fy = tf.keras.layers.Reshape((1, LATENT_DIM,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder

def build_conv2D_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(
        LATENT_DIM / 4 * LATENT_DIM / 4 * LATENT_DIM * 2, name='latent_dec_fy1')(fy)
    fy = tf.keras.layers.Reshape(
        (LATENT_DIM / 4, LATENT_DIM / 4, 2 * LATENT_DIM))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM, activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM / 2, name='conv_trans_y_2', activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM / 4, name='conv_trans_y_3', activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        1, name='conv_trans_y_4', activation='sigmoid', kernel_size=3, strides=(1, 1))(fy)
    fy = tf.keras.layers.Cropping2D(cropping=((5, 4), (4, 5)))(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(fy)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder

def build_conv2D_pointwise_encoder(custom_shape= INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_x_1')(x)
    fx = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fx)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1, 1),activation='relu', name='conv_x_2')(fx)
    fx = tf.keras.layers.Flatten()(fx)
    fx = tf.keras.layers.Dense(LATENT_DIM, name='latent_fx2')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_y_1')(y)
    fy = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fy)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1,1), activation='relu', name='conv_y_2')(fy)
    fy = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fy)
    fy = tf.keras.layers.Flatten()(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_fy2')(fy)
    fy = tf.keras.layers.Reshape((LATENT_DIM,1,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder

def build_dense_pointwise_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fh)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM,1,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fy1')(fy)

    y = tf.keras.layers.Dense(
        IMG_SIZE * IMG_SIZE, activation='relu', name='y_recon')(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder

def image_to_convnet(custom_shape = (IMG_SIZE, IMG_SIZE,1)):
    inputs = tf.keras.Input(shape = custom_shape, name = 'conv_input')
    x = tf.keras.layers.Conv2D(filters = LATENT_DIM,
                              kernel_size = 3,
                              strides = (2,2),
                              activation = 'relu',
                              name = 'conv_1')(inputs)
    x = tf.keras.layers.Conv2D(filters = 64,
                              kernel_size = 3,
                              strides = (2,2),
                              activation = 'relu',
                              name = 'conv_2')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_layer_1')(x)
    outputs = tf.keras.layers.Reshape((LATENT_DIM,1))(x)

    convnet = tf.keras.Model(inputs = inputs,
                            outputs = outputs,
                            name = 'conv_net_1')
    return convnet

def pos_to_dense(custom_shape = (1,3)):
    inputs = tf.keras.layers.Input(shape = custom_shape, name = 'dense_input')
    x = tf.keras.layers.Reshape(custom_shape)(inputs)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_1')(x)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_2')(x)
    outputs = tf.keras.layers.Reshape((LATENT_DIM,1))(x)

    densenet = tf.keras.Model(inputs = inputs, outputs = outputs, name = "dense_net")

    return densenet

def build_control_model():

    inputs = tf.keras.layers.Input(shape=(2,5))

    h = tf.keras.layers.Lambda(lambda x: x[:,0,:3])(inputs)
    p = tf.keras.layers.Lambda(lambda x: x[:,1,:])(inputs)

    h = tf.keras.layers.Reshape((1,3))(h)
    p = tf.keras.layers.Reshape((1,5))(p)

    fh = tf.keras.layers.Flatten()(h)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_3')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_4')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1))(fh)

    fp = tf.keras.layers.Flatten()(p)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_1')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_2')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_3')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_4')(fp)
    fp = tf.keras.layers.Reshape((LATENT_DIM,1))(fp)

    matmul = tf.keras.layers.Multiply()([fp, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_2')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_3')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_4')(fy)
    fy = tf.keras.layers.Dense(5, name = 'latent_y_out')(fy)
    fy = tf.keras.layers.Reshape((1,5))(fy)

    outputs = fy

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name='control_model')

    return model

def prepare_dataset(train_command, train_posture_before, train_posture_after, train_position_after, train_position_before):

    t_before = map(lambda x : x[0,:], train_posture_before)
    t_before = np.expand_dims(t_before, 1)

    t_command = map(lambda x : x[0,:], train_command)
    t_command = np.expand_dims(t_command, 1)

    t_visual_direction = normalize_vect(train_position_after - train_position_before)
    t_visual_direction = padding(t_visual_direction, 2)

    tmp_input = np.concatenate([t_visual_direction, t_before], axis = 1)

    train_control_dataset = (
        tf.data.Dataset.from_tensor_slices((tmp_input, t_command))
        .repeat(10)
        .shuffle(NB_DATA)
        .batch(BATCH_SIZE)
        )

    return train_control_dataset

def padding(t_visual, n):
    (a,b,c) = shape(t_visual)
    res = np.zeros((a,b,c+n))

    for i in range(len(t_visual)):

        res[i] = np.pad(t_visual[i][0], (0,n), 'constant')
    return res

def normalize_vect(visual_direction):
    res = np.zeros(shape(visual_direction))
    for i in range(len(visual_direction)):
        tmp = visual_direction[i]
        norm = np.linalg.norm(visual_direction[i])
        res[i]= tmp/norm
    return res

def test_visuomotor_control(control_model, current_posture,  visual_direction):
    postures = []
    postures.append(current_posture)
    posture = current_posture

    for i in range(400):

        vd = np.array([np.pad(visual_direction, (0,2), 'constant')])

        inputs = np.concatenate([vd, posture], axis = 0)
        inputs = np.expand_dims(inputs, 0)

        command = control_model.predict(inputs)

        posture = posture + command[0]
        posture = check_valid_posture(posture)
        postures.append(posture)

    return postures

def check_valid_posture(posture):
    valid_posture = np.zeros(shape(posture))

    if (np.abs(posture[0][0])>pi/2):
        valid_posture[0][0] = np.sign(posture[0][0])*pi/2
    else :
        valid_posture[0][0] = posture[0][0]

    if (np.abs(posture[0][1])>pi/2):
        valid_posture[0][1] = np.sign(posture[0][1])*pi/2
    else :
        valid_posture[0][1] = posture[0][1]

    if (np.abs(posture[0][2])>pi/2):
        valid_posture[0][2] = np.sign(posture[0][2])*pi/2
    else :
        valid_posture[0][2] = posture[0][2]

    if (np.abs(posture[0][3])>pi):
        valid_posture[0][3] = np.sign(posture[0][3])*pi
    else :
        valid_posture[0][3] = posture[0][3]

    if (np.abs(posture[0][4])>pi/2):
        valid_posture[0][4] = np.sign(posture[0][4])*pi/2
    else :
        valid_posture[0][4] = posture[0][4]

    return valid_posture

def plot_arm_from_posture(posture, target):
    phi1, phi2, theta1, psi1, psi2 = angles
    filename = 'images/%s.png' % time

    x = [0, 0, L1 * cos(phi1) * cos(phi2),
         (L1*cos(phi1)+L2*cos(theta1+phi1))*cos(phi2),
         (L1*cos(phi2)+L2*cos(theta1+phi2))*cos(phi1)+ L3*cos(psi1)*cos(psi2)]
    y = [0, 0, L1 * sin(phi1) * cos(phi2),
         (L1*cos(phi1)+L2*cos(theta1+phi1))*sin(phi2),
         (L1*cos(phi2)+L2*cos(theta1+phi2))*sin(phi1) + L3*cos(psi2)*sin(psi1)]
    z = [0, 0, L1 * sin(phi2),
         L1 * sin(phi1) + L2 * sin(theta1 + phi1)  ,
         L1 * sin(phi1) + L2 * sin(theta1 + phi1) + L3*sin(psi1)]

    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.3, right=0.3)
    ax.set_ylim(bottom=-0.3, top=0.3)
    ax.set_zlim(bottom=-0.3, top=0.3)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    close()

    return ax

def plot_elbow_from_posture(posture):
    phi1 = posture[0][0]
    phi2 = posture[0][1]
    theta1 = posture[0][2]

    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    x = [0, 0, L1 * cos(phi1) * cos(theta1)]
    y = [0, 0, L1 * cos(phi1) * sin(theta1)]  
    z = [0, 0, L1 * sin(phi1)]  
    ax.grid(False)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.2, right=0.2)
    ax.set_ylim(bottom=-0.2, top=0.2)
    ax.set_zlim(bottom=-0.2, top=0.2)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    ax.scatter(-0.5,-0.5,0.1, 'r')

def go_to_position(control_model, current_posture, target_position, nb_pass = 500):
    postures = []
    vd = []

    visual_direction = compute_vd_from_position(target_position, current_posture)

    postures.append(current_posture)
    vd.append((visual_direction))
    j = 0
    while (j < nb_pass) and (np.linalg.norm(target_position - np.array(control_robot(current_posture[0]))) > 0.1):

        inputs = np.expand_dims(np.concatenate([visual_direction, current_posture], axis=0), 0)

        new_command = control_model.predict(inputs)

        current_posture = current_posture + new_command[0]
        current_posture = check_valid_posture(current_posture)

        visual_direction = compute_vd_from_position(target_position, current_posture)

        postures.append(current_posture)
        vd.append((visual_direction))
        visual_direction = compute_vd_from_position(target_position, current_posture)/np.linalg.norm(visual_direction)
        j +=1

    return postures, np.array(vd)

def compute_vd_from_position(target_position, current_posture):

    current_position = control_robot(current_posture[0])
    tmp = (np.array(target_position)-np.array(current_position))
    return np.expand_dims(np.pad(tmp[0], (0,2), 'constant'), 0)

def is_distance_end_effector_to_target_ok(visual_direction):
    dx, dy, dz = visual_direction[0]

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    return (dist > 0.01)

def command_bornee(command):
    new_command = np.zeros(shape(command))
    for i in range(3):
        if command[0][0][i] > 2:
            new_command[0][0][i] = 2
        elif command[0][0][i] < -2:
            new_command[0][0][i] = -2
        else :
            new_command[0][0][i] = command[0][0][i]
        return new_command

def calcul_angular_error(position, direction_visuelle):
    return np.arccos(np.dot(position, direction_visuelle)/(np.linalg.norm(position)*np.linalg.norm(direction_visuelle)))
def calcul_position_error(position, target):

    return np.linalg.norm(position-target)

def get_end_effector_orientation(angles):
    phi1, phi2, theta1,psi1, psi2 = angles

    return np.array([psi1, psi2])

def prepare_dataset_with_orientation(train_command, train_posture_before, train_posture_after, train_position_after, train_position_before):

    t_before = map(lambda x : x[0,:], train_posture_before)
    t_before = np.expand_dims(t_before, 1)

    t_after = map(lambda x : x[0,:], train_posture_after)
    t_after = np.expand_dims(t_after, 1)

    t_command = map(lambda x : x[0,:], train_command)
    t_command = np.expand_dims(t_command, 1)

    orientation_before = np.array(map(lambda x : get_end_effector_orientation(x), t_before[:,0,:]))
    orientation_after = np.array(map(lambda x : get_end_effector_orientation(x), t_after[:,0,:]))

    t_orientation = orientation_after - orientation_before
    t_orientation = np.expand_dims(orientation, 1)

    t_visual_direction = normalize_vect(train_position_after - train_position_before)

    info = np.concatenate((t_visual_direction, t_orientation), axis=2)

    tmp_input = np.concatenate([info, t_before], axis = 1)

    train_control_dataset = (
        tf.data.Dataset.from_tensor_slices((tmp_input, t_command))
        .repeat(10)
        .shuffle(NB_DATA)
        .batch(100)
        )

    return train_control_dataset

def build_control_orientation_model():
    LATENT_DIM = 32
    inputs = tf.keras.layers.Input(shape=(2,5))

    h = tf.keras.layers.Lambda(lambda x: x[:,0,:])(inputs)
    p = tf.keras.layers.Lambda(lambda x: x[:,1,:])(inputs)

    h = tf.keras.layers.Reshape((1,5))(h)
    p = tf.keras.layers.Reshape((1,5))(p)

    fh = tf.keras.layers.Flatten()(h)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_3')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1))(fh)

    fp = tf.keras.layers.Flatten()(p)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_1')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_2')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_3')(fp)
    fp = tf.keras.layers.Reshape((LATENT_DIM,1))(fp)

    matmul = tf.keras.layers.Multiply()([fp, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_2')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_3')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_4')(fy)
    fy = tf.keras.layers.Dense(5, name = 'latent_y_out')(fy)
    fy = tf.keras.layers.Reshape((1,5))(fy)

    outputs = fy

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name='control_model')

    return model
EOF
