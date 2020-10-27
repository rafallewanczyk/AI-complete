

import tensorflow as tf
from tensorflow import keras
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                .format(num_groups))

        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

        self.conv_last      = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))              
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

def KerasShuffleNetV2_05(input_shape=(224, 224, 3)):
    image_input = keras.layers.Input(shape=input_shape)
    network = keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(image_input)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=False)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(24, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(24, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(48, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(96, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(1024, (1, 1), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=False)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.AveragePooling2D(pool_size=(7,7), strides=7, padding="same")(network)
    network = keras.layers.Dense(1000, input_shape=(1024,))(network) 

    model = keras.Model(inputs=image_input, outputs=network)

    return model

def KerasShuffleNetV2_10(input_shape=(224, 224, 3)):
    image_input = keras.layers.Input(shape=input_shape)
    network = keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(image_input)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=False)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(58, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(58, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(116, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(116, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (3, 3), strides=(2, 2), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(232, (3, 3), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)   
    network = keras.layers.Conv2D(232, (1, 1), strides=(1, 1), use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=True)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.Conv2D(1024, (1, 1), strides=(1, 1), padding="same", use_bias="FALSE")(network)
    network = keras.layers.BatchNormalization(axis=3, trainable=False, fused=False)(network)
    network = keras.layers.Activation("relu")(network)
    network = keras.layers.AveragePooling2D(pool_size=(7,7), strides=7, padding="same")(network)
    network = keras.layers.Dense(1000, input_shape=(1024,))(network) 

    model = keras.Model(inputs=image_input, outputs=network)

    return model

class PytorchToKeras(object):
    def __init__(self, pModel, kModel):
        super(PytorchToKeras, self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel
        tf.keras.backend.set_learning_phase(0)

    def __retrieve_k_layers(self):
        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):

        input = torch.randn(input_size)
        input = Variable(input.unsqueeze(0))
        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module, "weight"):
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self, input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i, (source_layer, target_layer) in enumerate(zip(self.__source_layers, self.__target_layers)):
            print(source_layer)
            weight_size = len(source_layer.weight.data.size())
            transpose_dims = []
            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

    def save_model(self, output_file):
        self.kModel.save(output_file)

    def save_weights(self, output_file):
        self.kModel.save_weights(output_file, save_format='h5')

pytorch_model = ShuffleNetV2()

pytorch_model.load_state_dict(torch.load('./shufflenetv2_x1_69.402_88.374.pth.tar'))

torch.save(pytorch_model, 'pt.pth')

pytorch_model = torch.load('pt.pth')

keras_model = KerasShuffleNetV2_10(input_shape=(224, 224, 3))

converter = PytorchToKeras(pytorch_model, keras_model)

converter.convert((3, 224, 224))

converter.save_model("keras.h5")

converter = tf.lite.TocoConverter.from_keras_model_file("keras.h5")

tflite_model = converter.convert()
open("tf.tflite", "wb").write(tflite_model)
EOF
