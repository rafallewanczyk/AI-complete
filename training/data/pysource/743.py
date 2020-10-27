import flask
from flask import request
import cv2
import numpy as np
import os
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
import math
import json
from json import JSONEncoder
from base64 import b64encode
from googletrans import Translator
from spellchecker import SpellChecker
import pytesseract
import enchant
from PyDictionary import PyDictionary
import nltk
from os.path import expanduser
from nltk.tag.stanford import StanfordPOSTagger
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = flask.Flask(__name__)
app.config["DEBUG"] = True
home = expanduser("~")
_path_to_model = home + '\stanford-postagger\models\english-bidirectional-distsim.tagger'
_path_to_jar = home + '\stanford-postagger\stanford-postagger.jar'
st = StanfordPOSTagger(_path_to_model, _path_to_jar)

def format_img_size(img):
    img_min_side = float(300)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img):
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    img /= img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img):
    img, ratio = format_img_size(img)
    img = format_img_channels(img)
    return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2

def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h

def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=3000):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    area = (x2 - x1) * (y2 - y1)

    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        area_union = area[i] + area[idxs[:last]] - area_int

        overlap = area_int / (area_union + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, dim_ordering, use_regr=True, max_boxes=3000, overlap_thresh=0.9):
    regr_layer = regr_layer / std_scaling

    anchor_sizes = anchor_box_scales
    anchor_ratios = anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'channels_first':
        (rows, cols) = rpn_layer.shape[2:]

    elif dim_ordering == 'channels_last':
        (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    if dim_ordering == 'channels_last':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    elif dim_ordering == 'channels_first':
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0]) / rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / rpn_stride
            if dim_ordering == 'channels_first':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))

            x, y = np.meshgrid(np.arange(cols), np.arange(rows))

            A[0, :, :, curr_layer] = x - anchor_x / 2
            A[1, :, :, curr_layer] = y - anchor_y / 2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result

def nn_base(input_tensor=None, trainable=False):
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    return x

def rpnn(base_layers, num_anchors):
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal',
                               name='rpn_conv1')(base_layers)

    x_class = tf.keras.layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                     name='rpn_out_class')(x)
    x_regr = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                                    name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifierr(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    input_shape = (num_rois, 7, 7, 512)

    pooling_regions = 7

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(out_roi_pool)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096, activation='relu', name='fc1'))(out)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(out)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4096, activation='relu', name='fc2'))(out)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5))(out)

    out_class = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
        name='dense_class_{}'.format(nb_classes))(out)
    out_regr = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
        name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

def build_model():
    print("Creating the model ")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Convolution2D(128, 3, 3, input_shape=(1, IMG_SIZE, IMG_SIZE), activation='relu',
                                            padding='same'))
    model.add(tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(tf.keras.layers.Convolution2D(256, 3, 3, activation='relu',  padding='same'))
    model.add(tf.keras.layers.Convolution2D(256, 3, 3, activation='relu',  padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),  padding='same'))

    model.add(tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Convolution2D(512, 3, 3, activation='relu',  padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),  padding='same'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(62))
    model.add(tf.keras.layers.Activation('softmax'))

    sgd = tf.keras.optimizers.SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def get_new_img_size(width, height, img_min_side=300):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height

def augment(img, augment=True):
    if augment:
        rows, cols = img.shape[:2]

        if use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)

        if use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)

        if rot_90:
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass
    return img

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele[0]
    return str1

class RoiPoolingConv(tf.keras.layers.Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = tf.keras.backend.image_data_format()
        assert self.dim_ordering in {'channels_last',
                                     'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)
        img = x[0]
        rois = x[1]

        input_shape = tf.keras.backend.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)
            num_pool_regions = self.pool_size

            if self.dim_ordering == 'channels_first':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = tf.keras.backend.cast(x1, 'int32')
                        x2 = tf.keras.backend.cast(x2, 'int32')
                        y1 = tf.keras.backend.cast(y1, 'int32')
                        y2 = tf.keras.backend.cast(y2, 'int32')

                        x2 = x1 + tf.keras.backend.maximum(1, x2 - x1)
                        y2 = y1 + tf.keras.backend.maximum(1, y2 - y1)

                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = tf.keras.backend.reshape(x_crop, new_shape)
                        pooled_val = tf.keras.backend.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'channels_last':
                x = tf.keras.backend.cast(x, 'int32')
                y = tf.keras.backend.cast(y, 'int32')
                w = tf.keras.backend.cast(w, 'int32')
                h = tf.keras.backend.cast(h, 'int32')
                rs = tf.image.resize(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)

        final_output = tf.keras.backend.concatenate(outputs, axis=0)
        final_output = tf.keras.backend.reshape(final_output,
                                                (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'channels_first':
            final_output = tf.keras.backend.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = tf.keras.backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def dilate(image, value):
    kernel = np.ones((value, value), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode(image, value):
    kernel = np.ones((value, value), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image, value):
    kernel = np.ones((value, value), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

PATH = "E:/FACULTATE/LICENTA/Licenta/"
anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1. / 1.41, 2. / 1.41], [2. / 1.41, 1. / 1.41]]
img_channel_mean = [103.939, 116.779, 123.68]
img_scaling_factor = 1.0
num_rois = 4
rpn_stride = 16
std_scaling = 4.0
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
num_features = 512
IMG_SIZE = 64
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a_small", "b_small",
              "c_small", "d_small", "e_small", "f_small", "g_small", "h_small", "i_small", "j_small", "k_small",
              "l_small", "m_small", "n_small", "o_small", "p_small", "q_small", "r_small", "s_small", "t_small",
              "u_small", "v_small", "w_small", "x_small", "y_small", "z_small"]
class_mapping = np.load(os.path.join(PATH, "Data/text_detection_class_mapping_info.npy"), allow_pickle=True)
class_mapping = np.array(class_mapping).tolist()
if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)
model_path_faster_rcnn = os.path.join(PATH, "Models/train-text-detection-weights-600.h5")
model_path_vgg = os.path.join(PATH, "Models/character-recognition-chars-natural-4.h5")
save_path = os.path.join(PATH, "Data/Results/")

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

img_input = tf.keras.layers.Input(shape=input_shape_img)
roi_input = tf.keras.layers.Input(shape=(num_rois, 4))
feature_map_input = tf.keras.layers.Input(shape=input_shape_features)

shared_layers = nn_base(img_input, trainable=True)
rpn_layers = rpnn(shared_layers, num_anchors)
classifier = classifierr(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = tf.keras.Model(img_input, rpn_layers)
model_classifier_only = tf.keras.Model([feature_map_input, roi_input], classifier)
model_classifier = tf.keras.Model([feature_map_input, roi_input], classifier)
model = build_model()
print('Loading weights from {}'.format(model_path_faster_rcnn))

model_rpn.load_weights(model_path_faster_rcnn, by_name=True)
model_classifier.load_weights(model_path_faster_rcnn, by_name=True)
model.load_weights(model_path_vgg)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

bbox_threshold = 0.8
translator = Translator()
languages = ['ar', 'bg', 'cs', 'da', 'nl', 'fr', 'de', 'el', 'hu', 'ja', 'pt', 'ro', 'es', 'tr']
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
d = enchant.Dict("en_US")
dictionary=PyDictionary()

def preProcess(image):
    image_pre_process = cv2.GaussianBlur(image, (5, 5), 0)
    image_pre_process = cv2.cvtColor(image_pre_process, cv2.COLOR_BGR2GRAY)
    image_pre_process = dilate(image_pre_process, 2)
    image_pre_process = erode(image_pre_process, 2)
    mean = np.average(image_pre_process)
    ret, mask = cv2.threshold(image_pre_process, mean, 255, cv2.THRESH_BINARY)
    image_final_process = cv2.bitwise_and(image_pre_process, image_pre_process, mask=mask)
    ret, new_img_process = cv2.threshold(image_final_process, mean, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image_pre_process = cv2.dilate(new_img_process, kernel, iterations=1)
    black_thresh = 100
    nblack = 0
    rows, cols = image_pre_process.shape
    for i in range(rows):
        for j in range(cols):
            k = image_pre_process[i, j]
            if k < black_thresh:
                nblack += 1
    n = rows * cols
    if (nblack > n / 2):
        image_pre_process = cv2.bitwise_not(image_pre_process)
    return image_pre_process

@app.route("/", methods=["POST"])
def upload():
    print("Receiving HTTP request...")
    content = request.files['file'].read()
    word = request.form.get('word')
    if word == "true":
        word = True
    else :
        word = False
    npimg = np.fromstring(content, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    cv2.imwrite('image.jpg', img)
    response = {}
    print("Image received - done")
    [data, data_translate, img, partsSpeech] = test(word)
    response['text'] = data
    response['speech'] = partsSpeech
    response['translate'] = data_translate
    b64_data = b64encode(cv2.imencode('.jpg', img)[1]).decode()
    response['image'] = b64_data
    print("Responding HTTP request...")
    return response

def test(word):
    global class_mapping
    print("Image processing starts: ")
    r = 0
    string_list = {}
    translate_list = {}
    img_original = cv2.imread('image.jpg')
    original_width = img_original.shape[1]
    original_height = img_original.shape[0]
    all_dets = []
    spell = SpellChecker(language="en", case_sensitive=True)
    img_original_copy = img_original.copy()

    print("Dimensions: ", img_original.shape)  
    if word:
        crop_img = img_original
        width_max = 2000
        height_max = 1000
        crop_img = cv2.resize(crop_img, (width_max, height_max), interpolation=cv2.INTER_CUBIC)
        gray_crop = preProcess(crop_img)
        image_new = gray_crop.copy()
        gray_crop = cv2.resize(gray_crop, (width_max - 2, height_max), interpolation=cv2.INTER_AREA)
        image_new[:, 0:width_max - 2] = gray_crop
        image_new[:, width_max - 2:width_max] = 255
        gray_crop = image_new

        cv2.imwrite(save_path + str(np.random.randint(0, 255)) + str(np.random.randint(0, 255)) + "lll.jpg", gray_crop)

        contours, hierarchy = cv2.findContours(gray_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        num_letters = 0

        height_word, width_word = gray_crop.shape
        area = width_word * height_word
        aspect_ratio = height_word / float(width_word)
        new_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            ar = h / float(w)
            if area * 0.01 < cv2.contourArea(ctr) < area * 0.75 and ar > 0.8 * aspect_ratio and h > 0.3 * height_word:
                num_letters += 1
                new_contours.append(ctr)

        roi = np.zeros((num_letters, 1, IMG_SIZE, IMG_SIZE), dtype=float)
        num_letters = 0

        for ctr in new_contours:
            x, y, w, h = cv2.boundingRect(ctr)
            dX = int(w * 0.05)
            dY = int(h * 0.05)
            min_x = max(x - 2 * dX, 0)
            min_y = max(y - 2 * dY, 0)
            max_x = min(x + w + 2 * dX, width_word)
            max_y = min(y + h + 2 * dY, height_word)

            letter = gray_crop[min_y:max_y, min_x:max_x]
            letter = dilate(letter, 7)
            letter = cv2.resize(letter, (64, 64), interpolation=cv2.INTER_AREA)

            cv2.imwrite(save_path + str(num_letters) + ".jpg", letter)

            roi[num_letters] = tf.expand_dims(letter, 0)
            num_letters += 1

        if len(roi) > 0:
            predictions = tf.argmax(model.predict(roi), axis=1)
            values = [CATEGORIES[i] for i in predictions]
            pred = model.predict(roi)
            pred_max = tf.math.reduce_max(pred, axis=1)
            values_copy = values.copy()
            for item in list(values):
                i = values_copy.index(item)
                if pred_max[i] < 0.4:
                    values.remove(item)
            string_list[r] = listToString(values)
            if not (all(map(str.isdigit, string_list[r]))):
                numbers = sum(c.isdigit() for c in string_list[r])
                lower_count = sum(map(str.islower, string_list[r]))
                upper_count = sum(map(str.isupper, string_list[r]))
                if numbers == 0 and len(string_list[r]) == 1:
                    string_list[r] = ""
                if numbers < lower_count + upper_count:
                    if len(string_list[r]) > 0:
                        firstLetter = string_list[r][0]
                        if string_list[r].lower() != spell.correction(string_list[r]):
                            string_list[r] = spell.correction(string_list[r])
                        if firstLetter.isupper() and lower_count > upper_count - 1:
                            string_list[r] = string_list[r].capitalize()
                        elif upper_count >= lower_count:
                            string_list[r] = string_list[r].upper()
                        else:
                            string_list[r] = string_list[r].lower()
        r += 1

    else:
        img = img_original
        x_img = augment(img, augment=False)
        (width, height) = (original_width, original_height)
        (rows, cols, _) = x_img.shape
        assert cols == width
        assert rows == height
        resized_height = height
        resized_width = width
        img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        X, ratio = format_img(img)

        if tf.keras.backend.image_data_format() == 'channels_last':
            X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, tf.keras.backend.image_data_format(), overlap_thresh=0.8)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // num_rois + 1):
            ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // num_rois:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= classifier_regr_std[0]
                    ty /= classifier_regr_std[1]
                    tw /= classifier_regr_std[2]
                    th /= classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([rpn_stride * x, rpn_stride * y, rpn_stride * (x + w), rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        if len(bboxes) == 0:
            print("No words found.")
            return [string_list, translate_list, img, {}]

        print("Words segmentation - done")

        keys = {}

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)
            new_boxes = np.array(sorted(new_boxes.tolist(), key=lambda box: (box[1], box[0])))

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                min_x = min(real_x1, real_x2)
                max_x = max(real_x1, real_x2)
                min_y = min(real_y2, real_y1)
                max_y = max(real_y2, real_y1)
                dX = int((max_x - min_x) * 0.02)
                dY = int((max_y - min_y) * 0.1)
                pad_min_x = max(min_x - 4*dX, 0)
                pad_min_y = max(min_y - 2*dY, 0)
                pad_max_x = min(max_x + 6*dX, resized_width)
                pad_max_y = min(max_y + 2*dY, resized_height)

                crop_img = img_original_copy[pad_min_y:pad_max_y, pad_min_x:pad_max_x]
                if pad_min_y > pad_max_y or pad_min_x > pad_max_x or pad_max_x > img.shape[1] or pad_max_y > img.shape[
                    0]:
                    continue
                cv2.imwrite(str(jk) + "result.jpg", crop_img)
                width_max = max(round(crop_img.shape[1]*original_width/resized_width), 1000)
                height_max = max(round(crop_img.shape[0]*original_height/resized_height), 1000)
                crop_img = cv2.resize(crop_img, (width_max, height_max), interpolation=cv2.INTER_CUBIC)

                gray_crop = preProcess(crop_img)

                gray_crop_erosed = erode(gray_crop, 3)
                contours, hierarchy = cv2.findContours(gray_crop_erosed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
                x, y, w, h = cv2.boundingRect(contours[0])
                gray_crop = gray_crop[y:y + h, x:x + w]

                image_new = gray_crop.copy()
                gray_crop = cv2.resize(gray_crop, (w - 2, h), interpolation=cv2.INTER_AREA)

                image_new[:, 0:w - 2] = gray_crop
                image_new[:, w - 2:w] = 255
                gray_crop = image_new
                cv2.imwrite(save_path + str(np.random.randint(0, 255)) + str(np.random.randint(0, 255)) + "lll.jpg",
                            gray_crop)

                contours, hierarchy = cv2.findContours(gray_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                num_letters = 0

                height_word, width_word = gray_crop.shape
                area = width_word * height_word
                aspect_ratio = height_word / float(width_word)
                new_contours = []
                for ctr in contours:
                    x, y, w, h = cv2.boundingRect(ctr)
                    ar = h / float(w)
                    if area*0.01 < cv2.contourArea(ctr) < area*0.75 and ar > aspect_ratio and h > 0.3*height_word :
                        min_x = max(x, 0)
                        min_y = max(y, 0)
                        max_x = min(x + w, width_word)
                        max_y = min(y + h, height_word)

                        letter = gray_crop[min_y:max_y, min_x:max_x]
                        black_thresh = 100
                        nblack = 0
                        rows, cols = letter.shape
                        for i in range(rows):
                            for j in range(cols):
                                k = letter[i, j]
                                if k < black_thresh:
                                    nblack += 1
                        n = rows * cols
                        if nblack < n / 4:
                            continue
                        new_contours.append(ctr)
                        num_letters += 1

                new_contours = sorted(new_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                roi = np.zeros((num_letters, 1, IMG_SIZE, IMG_SIZE), dtype=float)
                image_new = np.ones((64, 64), dtype=float) * 255
                num_letters = 0
                for ctr in new_contours:
                    x, y, w, h = cv2.boundingRect(ctr)
                    min_x = max(x, 0)
                    min_y = max(y, 0)
                    max_x = min(x + w, width_word)
                    max_y = min(y + h, height_word)
                    letter = gray_crop[min_y:max_y, min_x:max_x]

                    letter = dilate(letter, 5)
                    letter = cv2.resize(letter, (52, 52), interpolation=cv2.INTER_AREA)

                    image_new[5:57, 5:57] = letter
                    cv2.imwrite(save_path + str(jk) + str(num_letters) + ".jpg", image_new)
                    roi[num_letters] = tf.expand_dims(image_new, 0)
                    num_letters += 1
                if len(roi) > 0:
                    predictions = tf.argmax(model.predict(roi), axis=1)
                    values = [CATEGORIES[i] for i in predictions]
                    pred = model.predict(roi)
                    pred_max = tf.math.reduce_max(pred, axis=1)
                    values_copy = values.copy()

                    for item in list(values):
                        i = values_copy.index(item)
                        if pred_max[i] < 0.43:
                            values.remove(item)
                    string_list[r] = listToString(values)
                    keys[r] = jk
                    if not (all(map(str.isdigit, string_list[r]))):
                        numbers = sum(c.isdigit() for c in string_list[r])
                        lower_count = sum(map(str.islower, string_list[r]))
                        upper_count = sum(map(str.isupper, string_list[r]))
                        if numbers == 0 and len(string_list[r]) == 1:
                            string_list[r] = ""
                            continue
                        if numbers < lower_count + upper_count:
                            if len(string_list[r]) > 0:
                                firstLetter = string_list[r][0]
                                if string_list[r].upper() != spell.correction(string_list[r]).upper():
                                    string_list[r] = spell.correction(string_list[r])
                                if firstLetter.isupper() and lower_count > upper_count - 1:
                                    string_list[r] = string_list[r].capitalize()
                                elif upper_count >= lower_count:
                                    string_list[r] = string_list[r].upper()
                                else:
                                    string_list[r] = string_list[r].lower()

                r += 1
                all_dets.append((key, 100 * new_probs[jk]))

        print("Words recognition - done")

        storage = np.zeros((r, 4), dtype=float)
        for key in bboxes:
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)
            new_boxes = np.array(sorted(new_boxes.tolist(), key=lambda box: (box[1], box[0])))
            index = 0
            for jk in string_list.keys():
                (x1, y1, x2, y2) = new_boxes[int(keys[jk]), :]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                min_x = min(real_x1, real_x2)
                max_x = max(real_x1, real_x2)
                min_y = min(real_y2, real_y1)
                max_y = max(real_y2, real_y1)
                dX = int((max_x - min_x) * 0.02)
                dY = int((max_y - min_y) * 0.1)
                pad_min_x = max(min_x - 4 * dX, 0)
                pad_min_y = max(min_y - 2 * dY, 0)
                pad_max_x = min(max_x + 6 * dX, resized_width)
                pad_max_y = min(max_y + 2 * dY, resized_height)
                if len(string_list[jk]) <= 0:
                    continue
                storage[index] = [pad_min_x,pad_min_y, pad_max_x, pad_max_y]
                index += 1
                cv2.rectangle(img_original_copy, (pad_min_x, pad_min_y), (pad_max_x, pad_max_y), (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)

    text = ""
    string_result = {}
    translate_result = {}
    index = -1
    if len(string_list) > 0:
        for iii in string_list:
            index += 1
            if len(string_list[iii]) <= 0:
                continue
            if iii > 0 and storage[index][1] > storage[index - 1][1]:
                text = text + "\n"
            translate_list[iii] = translator.translate(string_list[iii], src="en", dest="ro").text
            text = text + string_list[iii] + " "
    string_result[0] = text
    if len(text) > 0:
        for lan in languages:
            translate_result[lan] = translator.translate(text, src="en", dest=lan).text

    result = st.tag(text.split())
    dictOfWords = { i : result[i] for i in range(0, len(result) ) }

    if not word:
        print("Probability of localization for founded words: ")
        for i in string_list:
            print("\t", string_list[i], all_dets[i])
    else:
        print("Word recognition - done")
    img_original_copy = cv2.resize(img_original_copy, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("result.jpg", img_original_copy)

    return [string_result, translate_result, img_original_copy, dictOfWords]

if __name__ == '__main__':
    app.run(host='192.168.100.5', port='8080')
EOF
