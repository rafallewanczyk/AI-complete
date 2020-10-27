

import pydicom
import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications import inception_resnet_v2
from keras.applications.inception_v3 import preprocess_input
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from collections import defaultdict
import random
from random import randint
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)+ 1e-7
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)+ 1e-7
    TP = K.sum(y_pred * y_true)
    return TP/P
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))+ 1e-7
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))+ 1e-7
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto(allow_soft_placement=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess = tf.Session(config=config)

config = tf.ConfigProto()
import json
import os
from pycocotools.mask import *
file=os.listdir("../data/jinnan2_round2_train_20190401/jinnan2_round2_train_20190401/normal/")
with open('../data/jinnan2_round2_train_20190401/jinnan2_round2_train_20190401/train_restriction.json','r') as load_f:
    load_dict = json.load(load_f)
pr="../data/jinnan2_round2_train_20190401/jinnan2_round2_train_20190401/restricted/"   
pn="../data/jinnan2_round2_train_20190401/jinnan2_round2_train_20190401/normal/" 
images=defaultdict()

for i in load_dict['images']:
    images[i['id']]=i
ids=list(images.keys())
n=len(ids)
masks=defaultdict()
for j in ids:
    l=[]
    for i in load_dict['annotations']:
        if i['image_id']==j:
            l.append(i)
    masks[j]=l
count=0
z=ids.copy()
for i in z:
    if len(masks[i])==0:
        print(i)
        ids.remove(i)

x_train, X_valid = train_test_split(ids, test_size=0.1,random_state=3)

n_train, n_valid = train_test_split(file, test_size=0.1,random_state=3)
width = 512

xtrain=[x_train,n_train]
xvalid=[X_valid,n_valid]
ltr=len(x_train)+len(n_train)
lv=len(X_valid)+len(n_valid)

datagen = ImageDataGenerator(
			rotation_range=90,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.01,
			zoom_range=0.1,         
			horizontal_flip=True,
			vertical_flip=False,
			fill_mode='constant',cval=0,
			)
datagenx = ImageDataGenerator(
			rotation_range=30,
			width_shift_range=0.01,
			height_shift_range=0.01,
			shear_range=0.01,
			zoom_range=[0.8,1.2],         
			horizontal_flip=True,
			vertical_flip=False,
			fill_mode='constant',cval=0,
			)
datagen1 = ImageDataGenerator(
			rotation_range=0,
			width_shift_range=0,
			height_shift_range=0,
			shear_range=0,
			zoom_range=0,         
			horizontal_flip=False,
			vertical_flip=False,
			fill_mode='nearest'
			)
def rcrop(img,width):
    h,w=img.shape[0],img.shape[1]
    if h <width and w<width:
        nimg=np.zeros([width,width,img.shape[2]])
        nimg[:img.shape[0],:img.shape[1],:]=img
        return nimg
    if w<width:
        nimg=np.zeros([img.shape[0],width,img.shape[2]])
        nimg[:img.shape[0],:img.shape[1],:]=img      
        img=nimg
        w=width
    if h<width:
        nimg=np.zeros([width,img.shape[1],img.shape[2]])
        nimg[:img.shape[0],:img.shape[1],:]=img  
        img=nimg
        h=width
    x=randint(0,h-width)
    y=randint(0,w-width)
    return img[x:x+width,y:y+width,:]
def my_datagen(path,size=16,width = 299,s=None):
    l0,l1=path
    v0=list(range(len(l0)))
    v1=list(range(len(l1)))
    np.random.shuffle(v0)
    np.random.shuffle(v1)	
    i = 0
    i0=0
    i1=0
    add=1
    while True:
        X,Yn,Y = [],[],[]
        MY=[]
        A,S=[],[]
        X1=[]
        MY1=[]
        MY2=[]
        lm1,lm2,lm3,lm4,lm5=[],[],[],[],[]
        zm=[]
        yn=0
        while len(X)<size:
                if i%2==0:
                    flag=0
                    pa=l0[v0[i0]]
                    i0=i0+1
                    if i0 >= len(l0):
                        i0=i0-len(l0)
                        np.random.shuffle(v0)
                if i%2!=0:
                    flag=1
                    pa=l1[v1[i1]]
                    i1=i1+1
                    if i1 >= len(l1):
                        i1=i1-len(l1)
                        np.random.shuffle(v1)
                if flag==1:
                    imgc=cv2.imread(pn+pa)
                if flag==0:
                    imgc=cv2.imread(pr+images[pa]['file_name'])
                    mask=masks[pa]

                imgc=np.asarray(imgc,dtype='uint8')
                a=max(imgc.shape[0],imgc.shape[1])

                img0=np.zeros([imgc.shape[0],imgc.shape[1],8])
                kn=np.zeros([imgc.shape[0],imgc.shape[1],5])
                kn0=np.zeros([imgc.shape[0],imgc.shape[1],5])
                addf=randint(3,10)
                if addf>2:
                    if flag==0:
                        for j in mask:
                            lost=randint(0,2)
                            maskp=decode(frPyObjects(j['segmentation'],imgc.shape[0],imgc.shape[1]))
                            c=j['category_id']-1
                            if lost ==2 and maskp.sum()>0:
                                newm=maskp.copy()
                                newm=newm.sum(axis=2)
                                imgc[:,:,0][maskp.reshape(imgc.shape[0],imgc.shape[1])==1]=255
                                imgc[:,:,1][maskp.reshape(imgc.shape[0],imgc.shape[1])==1]=255
                                imgc[:,:,2][maskp.reshape(imgc.shape[0],imgc.shape[1])==1]=255
                                kn[:imgc.shape[0],:imgc.shape[1],c]=kn[:imgc.shape[0],:imgc.shape[1],c]-maskp.reshape(imgc.shape[0],imgc.shape[1])
                                kn[kn<0]=0
                            if lost<2:
                                kn[:imgc.shape[0],:imgc.shape[1],c]=kn[:imgc.shape[0],:imgc.shape[1],c]+maskp.reshape(imgc.shape[0],imgc.shape[1])
                    if flag==1 or addf>5:
                        ids=np.random.choice(x_train, 1)[0]
                        addi=cv2.imread(pr+images[ids]['file_name'])
                        mask=masks[ids]
                        for j in mask:
                            lost=randint(1,2)
                            maskp=decode(frPyObjects(j['segmentation'],addi.shape[0],addi.shape[1]))
                            c=j['category_id']-1
                            if lost ==2 and maskp.sum()>0:
                                addpart=maskp*addi
                                addpart = addpart[np.newaxis, :]
                                x = datagen.flow(addpart,batch_size=1).next()
                                if x.sum()==0:
                                    continue
                                x=np.asarray(x,dtype='uint8')
                                addpart=x[0]
                                newm=addpart.copy()
                                newm=newm.sum(axis=2)
                                newm[newm>0]=1
                                xs=newm.sum(axis=1)
                                xs[xs>0]=1
                                if 1 not in list(xs):
                                    continue
                                x0=list(xs).index(1)
                                xl=xs.sum()
                                xs=newm.sum(axis=0)
                                xs[xs>0]=1
                                y0=list(xs).index(1)
                                yl=xs.sum()
                                if imgc.shape[0]-xl<20 or imgc.shape[1]-yl<20:
                                    continue
                                nx=randint(10,imgc.shape[0]-xl-10)
                                ny=randint(10,imgc.shape[1]-yl-10)
                                newm[newm!=0]=1
                                x0=int(x0)
                                y0=int(y0)
                                xl=int(xl)
                                yl=int(yl)
                                nk=addpart[x0:x0+xl,y0:y0+yl,:]
                                jk=newm[x0:x0+xl,y0:y0+yl].reshape(xl,yl,1)
                                z=randint(5,10)
                                z=1
                                qi=imgc[nx:nx+xl,ny:ny+yl,:]
                                nk1=(1-jk)*qi+z*(jk*nk)+(1-z)*(jk*qi)
                                nk1=nk1.astype('uint8')
                                imgc[nx:nx+xl,ny:ny+yl,:]=nk1
                                kn[nx:nx+xl,ny:ny+yl,c]=kn[nx:nx+xl,ny:ny+yl,c]+newm[x0:x0+xl,y0:y0+yl]
                if addf<=2:
                    if flag==0:
                        for j in mask:
                            maskp=decode(frPyObjects(j['segmentation'],imgc.shape[0],imgc.shape[1]))
                            c=j['category_id']-1
                            kn[:imgc.shape[0],:imgc.shape[1],c]=kn[:imgc.shape[0],:imgc.shape[1],c]+maskp.reshape(imgc.shape[0],imgc.shape[1])

                kn0[:imgc.shape[0],:imgc.shape[1],:]=kn
                kn=kn0
                img0[:imgc.shape[0],:imgc.shape[1],:3]=imgc          

                img0[:,:,3:]=kn
                img0=rcrop(img0,width)

                img0=np.asarray(img0,dtype='uint8')
                img = img0[np.newaxis, :]
                x = datagenx.flow(img,batch_size=1).next()
                x=np.asarray(x,dtype='uint8')
                x=x[0]

                kn=x[:,:,3:].copy()
                x=x[:,:,:3]
                X.append(x.reshape([width,width,3]))
                MY.append(kn.reshape([width,width,5]))
                kn[kn>1]=1
                kn[kn!=1]=0
                MY1.append(kn.reshape([width,width,5]))
                i =i+ 1
                yn=yn+1
                if i >= len(l0)+len(l1):
                    i=i-len(l0)-len(l1)
        yield np.array(X),[np.array(MY1),np.array(MY)]
def my_datagen1(path,size=16,width = 299,s=None):
    l0,l1=path
    v0=list(range(len(l0)))
    v1=list(range(len(l1)))
    np.random.shuffle(v0)
    np.random.shuffle(v1)	
    i = 0
    i0=0
    i1=0
    add=1
    while True:
        X,Yn,Y = [],[],[]
        MY=[]
        A,S=[],[]
        X1=[]
        MY1=[]
        MY2=[]
        yn=0
        while len(X)<size:
                if i%2==0:
                    flag=0
                    pa=l0[v0[i0]]
                    i0=i0+1
                    if i0 >= len(l0):
                        i0=i0-len(l0)
                        np.random.shuffle(v0)
                if i%2!=0:
                    flag=1
                    pa=l1[v1[i1]]
                    i1=i1+1
                    if i1 >= len(l1):
                        i1=i1-len(l1)
                        np.random.shuffle(v1)
                if flag==1:
                    imgc=cv2.imread(pn+pa)
                if flag==0:
                    imgc=cv2.imread(pr+images[pa]['file_name'])
                    mask=masks[pa]
                imgc=np.asarray(imgc,dtype='uint8')
                a=max(imgc.shape[0],imgc.shape[1])
                img0=np.zeros([a,a,8])
                kn=np.zeros([a,a,5])
                if add>0:
                    img0=np.zeros([imgc.shape[0],imgc.shape[1],8])
                    kn=np.zeros([imgc.shape[0],imgc.shape[1],5])                    
                if flag==0:
                    for j in mask:
                        maskp=decode(frPyObjects(j['segmentation'],imgc.shape[0],imgc.shape[1]))
                        c=j['category_id']-1
                        kn[:imgc.shape[0],:imgc.shape[1],c]=kn[:imgc.shape[0],:imgc.shape[1],c]+maskp.reshape(imgc.shape[0],imgc.shape[1])

                img0[:imgc.shape[0],:imgc.shape[1],:3]=imgc          

                img0[:,:,3:]=kn
                if add==0:
                    img0=resize(img0, (width,width), mode='reflect')
                if add>0:
                    img0=rcrop(img0,width)
                img0=np.asarray(img0,dtype='uint8')
                img = img0[np.newaxis, :]
                x = datagen1.flow(img,batch_size=1).next()
                x=np.asarray(x,dtype='uint8')
                x=x[0]

                kn=x[:,:,3:].copy()
                x=x[:,:,:3]
                X.append(x.reshape([width,width,3]))
                MY.append(kn.reshape([width,width,5]))

                kn[kn>1]=1
                kn[kn!=1]=0
                MY1.append(kn.reshape([width,width,5]))
                i =i+ 1
                yn=yn+1
                if i >= len(l0)+len(l1):
                    i=i-len(l0)-len(l1)
        yield np.array(X),[np.array(MY1),np.array(MY)]

config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications import inception_resnet_v2
from keras.applications.inception_v3 import preprocess_input
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications import inception_resnet_v2
from keras.applications.inception_v3 import preprocess_input
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import load_model

from model import Deeplabv3
input_tensor = Input((width, width, 3))
x = input_tensor

modelb = Deeplabv3(input_tensor=x,input_shape=(width, width, 3),backbone='xception', classes=5) 
x=modelb.layers[-2].output
x1= keras.layers.Conv2D(5,1, name='cf')(x)
outputs = keras.layers.Conv2D(5,1, activation='sigmoid',name='cc')(x)
model=Model(input_tensor,[outputs,x1] )

model.summary()

X=[]

loss=list()
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

def box_loss(y_true, y_pred):
    a=(-1)/24
    b=a
    val = np.array([[a,a,a,a,a],[a,b,b,b,a], [a,b,1,b,a],[a,b,b,b,a],[a,a,a,a,a]])
    kv =K.reshape( K.variable(value=val),(5,5,1,1))
    km=K.conv2d(y_true, kv, strides=(1,1), padding='same')
    km=K.abs(km)
    km=km+1
    a=(1)/24
    b=a
    val1 = np.array([[a,a,a,a,a],[a,b,b,b,a], [a,b,0.,b,a],[a,b,b,b,a],[a,a,a,a,a]])
    kv1 =K.reshape( K.variable(value=val1),(5,5,1,1))
    km1=K.conv2d(y_true, kv1, strides=(1,1), padding='same')
    km1=km1+1
    l=K.binary_crossentropy(y_pred, y_true, from_logits=False)
    l1=l*km
    l2=l*km1
    com=K.mean(l1)-K.mean(l2)
    c1=tf.ones_like(l1)*com
    ll= tf.where(K.greater(c1,tf.zeros_like(l1)), l1, l2)
    return l2

def box_f_loss(y_true, y_pred):
    a=(-1)/24
    val = np.array([[a,a,a,a,a],[a,a,a,a,a], [a,a,1,a,a],[a,a,a,a,a],[a,a,a,a,a]])
    kv =K.reshape( K.variable(value=val),(5,5,1,1))
    km=K.conv2d(y_true, kv, strides=(1,1), padding='same')
    km=K.abs(km)
    km=km+1
    a=(1)/24
    val = np.array([[a,a,a,a,a],[a,a,a,a,a], [a,a,0.,a,a],[a,a,a,a,a],[a,a,a,a,a]])
    kv =K.reshape( K.variable(value=val),(5,5,1,1))
    km1=K.conv2d(y_true, kv, strides=(1,1), padding='same')
    km1=km1+1

    gamma=2.
    alpha=0.25
    pt_1 = tf.where(K.greater(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    l1=-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)*km)-K.sum(km*(1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    l2=-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)*km1)-K.sum(km1*(1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    com=K.mean(l1)-K.mean(l2)
    c1=tf.ones_like(l1)*com
    ll= tf.where(K.greater(c1,tf.zeros_like(l1)), l1, l2)
    return l2
smooth=1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def sum_loss(y_true, y_pred):

    l=K.binary_crossentropy(y_pred, y_true, from_logits=False)
    p1=tf.where(K.greater(y_pred,0.5), y_pred, tf.zeros_like(y_pred))
    p0=tf.where(K.greater(y_true,0.5),  tf.zeros_like(y_pred),y_pred)
    p2=tf.where(K.greater(y_true,0.5), y_pred, tf.zeros_like(y_pred))
    l1=tf.reduce_sum(y_true, axis=[1, 2, 3])-tf.reduce_sum(p1, axis=[1, 2, 3])
    l2=tf.reduce_sum(p0, axis=[1, 2, 3])
    l3=tf.reduce_sum(y_true, axis=[1, 2, 3])-tf.reduce_sum(p2, axis=[1, 2, 3])
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    l5=1-score

    ll=l1+l2+l3+l5
    return ll
def size_loss(y_true, y_pred):

    l=K.binary_crossentropy(y_pred, y_true, from_logits=False)

    l4=keras.layers.multiply([tf.reduce_sum(tf.ones_like(y_true), axis=[1, 2, 3])/(0.5*tf.reduce_sum(tf.ones_like(y_true), axis=[1, 2, 3])+tf.reduce_sum(y_true, axis=[1, 2, 3])),l])

    return l4
for i in range(1):
    loss.append(focal_loss(alpha=0.25, gamma=2))
checkpointer = ModelCheckpoint(filepath='../model/ofxdn14s.best.h5', verbose=1, monitor='val_loss',mode='min',save_weights_only=True,
                               save_best_only=True)

def mean_iou4(y_true, y_pred):
    y_pred = tf.where(K.greater(y_pred, 0.3), tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
def mean_iou6(y_true, y_pred):
    y_pred = tf.where(K.greater(y_pred, 0.4), tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
lr = 1e-5
size =12
model.load_weights('../model/x3.best.h5')

adam = Adam(lr=lr, beta_2=0.9999, epsilon=1e-08)

for i in model.layers[:356]:
    i.trainable=False
model.compile(optimizer=adam, loss=[focal_loss(alpha=0.25, gamma=2),'mean_absolute_error'],metrics={'cc':['accuracy',mean_iou,mean_iou4,mean_iou6],'cf':['accuracy',mean_iou]})

model.summary()

h=model.fit_generator(
		my_datagen(xtrain,size=size,width=width),
		steps_per_epoch=ltr//size, 
		validation_data=my_datagen1(xvalid,size=1,width=width),
		validation_steps=lv*3,
		epochs=3,
		callbacks=[EarlyStopping(patience=3), checkpointer],
		workers=-1,
		verbose=1)

lr = 1e-5
size=3

for i in model.layers[:356]:
    i.trainable=True
adam = Adam(lr=lr, beta_2=0.9999, epsilon=1e-08)
checkpointer = ModelCheckpoint(filepath='../model/x4.best.h5', verbose=1, save_weights_only=True,monitor='val_cc_mean_iou',mode='max',
                               save_best_only=True)

model.compile(optimizer=adam, loss=['binary_crossentropy','mean_absolute_error'],metrics={'cc':['accuracy',mean_iou,mean_iou4,mean_iou6],'cf':['accuracy',mean_iou]})
model.summary()

X=[]

h=model.fit_generator(
		my_datagen(xtrain,size=size,width=width),
		steps_per_epoch=ltr//size, 
		validation_data=my_datagen1(xvalid,size=1,width=width),
		validation_steps=lv*3,
		epochs=60,
		callbacks=[EarlyStopping(patience=3), checkpointer],
		workers=-1,
		verbose=1)

lr = 1e-6

adam = Adam(lr=lr)
X=[]

checkpointer = ModelCheckpoint(filepath='../model/x4.best.h5', verbose=1, save_weights_only=True,monitor='val_cc_mean_iou',mode='max',
                               save_best_only=True)
model.compile(optimizer=adam, loss=['binary_crossentropy','mean_absolute_error'],metrics=['accuracy',mean_iou])

h=model.fit_generator(
		my_datagen(xtrain,size=size,width=width),
		steps_per_epoch=ltr//size, 
		validation_data=my_datagen1(xvalid,size=1,width=width),
		validation_steps=lv*3,
		epochs=60,
		callbacks=[EarlyStopping(patience=9), checkpointer],
		workers=-1,
		verbose=1)

        
EOF
