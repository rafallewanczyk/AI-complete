import tensorflow as tf
import numpy as np, os, sys, re
import random, json, string, pickle
import keras
from keras.models import Sequential
from keras.models import Model
import keras.layers
from keras.layers import Merge, Dense, merge
import keras.models
from keras.optimizers import SGD
import keras.callbacks
from keras.preprocessing import image

img_names = []
encoding = []
file_path = os.path.dirname(os.path.abspath(__file__)) + "/pubfig_imgs"
with open("pubfig_attributes.txt") as img_file:
    count = 0
    print("Starting to fetch attributes")
    count = 0
    for line in img_file:
        arr = re.split(' |\t', line)
        if arr[0] == '
            continue
        img_name = arr[0]
        num = 0
        for i in range(1,20):
            try:
                test = int(arr[i])
                img_name += '-'+arr[i]
                num = i
                break
            except:
                img_name += '-'+arr[i]
        img_name += ".jpg"
        img_attr = arr[num:]
        for i in range(len(img_attr)):
            item = float(img_attr[i])
            if item <= 0:
                img_attr[i] = 0
            else:
                img_attr[i] = 1
        img_names.append(img_name)
        encoding.append(img_attr)
        count += 1
        if count % 10000 == 0:
            print("10000 more done")

encoding = np.asarray(encoding)
img_names = np.asarray(img_names)

concatenated_vector = keras.layers.Input(shape=(4, 4, 128))

gen1_input = keras.layers.Input(shape=(4, 4, 1152))
di1_input = keras.layers.Input(shape=(128, 128, 3))
real_image = keras.layers.Input(shape=(128, 128, 3))

di2_input = keras.layers.Input(shape=(256, 256, 6))
gen2_input = keras.layers.Input(shape=(128, 128, 3))
main_model = Sequential()
full_model = Sequential()

s1_gen = Sequential()
s1_dis = Sequential()
s1_di = Sequential()
s1_c = Sequential()

s2_gen = Sequential()
s2_dis = Sequential()

train_model_s1 = Sequential()
train_model_s2 = Sequential()

train_model_full = Sequential()

text_input = Sequential()
text_input.add(keras.layers.convolutional.Conv2D(128, 1, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(4, 4, 128)))

s1_gen = keras.layers.convolutional.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(4, 4, 1152))(gen1_input)
s1_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
gen1_output = keras.layers.convolutional.Conv2D(3, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_generator = Model(inputs=[gen1_input], outputs=[gen1_output])

s1_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(128, 128, 3))(di1_input)
s1_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
di1_output = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='inter')(s1_di)

s1_c = merge([di1_output, concatenated_vector], mode='concat',)
s1_c = keras.layers.core.Reshape((1, 18432))(s1_c)
s1_c = keras.layers.core.Dense(1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_c)

s1_discriminator = Model(inputs=[di1_input, concatenated_vector], outputs=[s1_c])
s1_discriminator.summary()

s1 = s1_discriminator([s1_generator(gen1_input), concatenated_vector])
train_model_s1 = Model(inputs=[gen1_input, concatenated_vector], outputs=[s1])

d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
s1_generator.compile(loss='binary_crossentropy', optimizer="SGD")
train_model_s1.compile(
loss='binary_crossentropy', optimizer=g_optim)
s1_discriminator.trainable = True
s1_discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

for n in range(294, 304):
    img_path = img_names[n] 
    try:
        img = image.load_img(os.path.join("./pubfig_resized/",img_path), target_size=(128, 128))
    except Exception as e:
        print(e)
        print(os.path.join("./pubfig_resized/",img_path))
        continue
    img = image.img_to_array(img)
    img = img.reshape(1, 128, 128, 3)
    print(encoding[n])
    conc = np.concatenate([encoding[n], np.zeros(54)])
    print(conc.shape)
    enc = np.tile(conc,16)
    print(enc.shape)
    enc = enc.reshape((1,4,4,128)) 
    noise = np.random.normal(size=(1,4,4,1024)) 
    generator_input = np.concatenate((noise, enc), axis=3)

    print(generator_input.shape, type(generator_input), gen1_input.shape, type(gen1_input))
    synth_img = s1_generator.predict(x=generator_input) 
    both_img = np.concatenate((img, synth_img), axis=0)

    gt1 = np.array(1)
    gt2 = np.array(0)
    gt1 = gt1.reshape(1,1,1)
    gt2 = gt2.reshape(1,1,1)
    both_gt = np.concatenate((gt1, gt2))
    both_enc = np.concatenate((enc, enc))

    s1_discriminator.train_on_batch(x=[both_img, both_enc], y=[both_gt])

    new_noise = np.random.normal(size=(1,4,4,1024)) 
    new_noise = np.concatenate((new_noise, enc), axis=3)

    gt3 = np.array(0)   
    gt3 = gt3.reshape(1,1,1)
    train_model_s1.train_on_batch(x=[new_noise, enc], y=gt3)


EOF
