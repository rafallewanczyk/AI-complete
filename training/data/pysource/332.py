

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import skimage.color
import skimage.io
import skimage.transform

from mmid.resnet50 import ResNet50
from mmid.effnet import EfficientNet
from mmid import utils

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

catalog_path = "/workspace/Cascade_RCNN/data/train_data/catalog/"
instance_path = "/workspace/Cascade_RCNN/data/train_data/instance/"
val_catalog_path = "/workspace/Cascade_RCNN/data/test_data/catalog/"
val_instance_path = "/workspace/Cascade_RCNN/data/test_data/instance/"

def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)

def load_mmid_image(dataset_path, config, asin, mode, augment=False, augmentation=None):
    image_dir = os.path.join(dataset_path, asin)
    if dataset_path == instance_path or dataset_path == val_instance_path:
    	image_dir = os.path.join(image_dir, "ASIN")
    if dataset_path == catalog_path or dataset_path == val_catalog_path:
        image_dir = os.path.join(image_dir, "catalog")
    if mode == 'train' or mode == 'val':
        image_file = random.choice(os.listdir(image_dir))
    else:
        image_file = os.listdir(image_dir)[0]
    image_path = os.path.join(image_dir, image_file)
    image = skimage.io.imread(image_path)

    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    if mode == 'train'or mode == 'val':
        rotation = [0, 90, 180, 270]
        degree = random.choice(rotation)
        image = skimage.transform.rotate(image, degree)

    return image

def mmid_data_generator_rnd(config,  augment=False, augmentation=None,
                   batch_size=1, no_augmentation_sources=None, mode='train'):
    b = 0  
    image_index = -1
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    while True:
        try:
            asins = os.listdir(catalog_path)
            image_index = (image_index + 1)
            if image_index == 0:
              np.random.shuffle(asins)
            asin = asins[image_index]
            catalog_image = load_mmid_image(catalog_path, config, asin, mode, augment=augment, augmentation=None)
            instance_image = load_mmid_image(instance_path, config, asin, mode, augment=augment, augmentation=None)

            if b == 0:
                catalog_batch_images = np.zeros(
                    (batch_size,) + catalog_image.shape, dtype=np.float32)

                instance_batch_images = np.zeros(
                    (batch_size,) + instance_image.shape, dtype=np.float32)
	    
            catalog_batch_images[b] = catalog_image
            instance_batch_images[b] = instance_image
            b += 1

            if b >= batch_size:
                inputs = [catalog_batch_images, instance_batch_images]
                outputs = []

                yield inputs, outputs

                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            logging.exception("Error processing image {}".format(asin))
            error_count += 1
            if error_count > 5:
                raise

class mmid_data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, config,  augment=False, augmentation=None,
                   batch_size=1, no_augmentation_sources=None, mode='train'):
        'Initialization'
        self.config = config
        self.batch_size = batch_size
        self.shuffle = True
        self.mode = mode
        if self.mode == 'train':
           self.ASINs = os.listdir(catalog_path)
           self.catalog_path = catalog_path
           self.instance_path = instance_path
        else:
           self.ASINs = os.listdir(val_catalog_path)
           self.catalog_path = val_catalog_path
           self.instance_path = val_instance_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ASINs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        ASINs_temp = [self.ASINs[k] for k in indexes]

        inputs, outputs = self.__data_generation(ASINs_temp)

        return inputs, outputs

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ASINs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ASINs_temp):
        'Generates data containing batch_size samples'

        for i, ASIN in enumerate(ASINs_temp):
            catalog_image = load_mmid_image(catalog_path, self.config, ASIN, self.mode, augment=False, augmentation=None)
            instance_image = load_mmid_image(instance_path, self.config, ASIN, self.mode, augment=False, augmentation=None)

            if i == 0:
                catalog_batch_images = np.zeros(
                    (self.batch_size,) + catalog_image.shape, dtype=np.float32)

                instance_batch_images = np.zeros(
                    (self.batch_size,) + instance_image.shape, dtype=np.float32)

            catalog_batch_images[i] = catalog_image
            instance_batch_images[i] = instance_image

        inputs = [catalog_batch_images, instance_batch_images]
        outputs = []

        return inputs, outputs

class MMID_vec():

    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        assert mode in ['training', 'inference']
        config.MODE = mode 

        h, w = config.IMAGE_SHAPE[:2]
        catalog_input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="catalog_input_image")

        instance_input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="instance_input_image")

        if config.BACKBONE == 'resnet':
           resnet_model_fc = ResNet50(include_top=False, weights='imagenet', name='resnet50_fc',
                                      input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3), pooling=None)
           resnet_model_wb = ResNet50(include_top=False, weights='imagenet', name='resnet50_wb',
                                      input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3), pooling=None)
           embedding_wb = resnet_model_wb(catalog_input_image)
           embedding_fc = resnet_model_fc(instance_input_image)
        elif config.BACKBONE == 'effnet':
           effnet_model_fc = EfficientNet(1.4, 1.8, 380, 0.4,
                                          model_name='efficientnet-b4',
                                          include_top=False,
                                          input_tensor=None, 
                                          input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3),
                                          pooling=None,
                                          post_fix='1')
           effnet_model_wb = EfficientNet(1.4, 1.8, 380, 0.4,
                                          model_name='efficientnet-b4',
                                          include_top=False,
                                          input_tensor=None, 
                                          input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3),
                                          pooling=None,
                                          post_fix='2')

           embedding_wb = effnet_model_wb(catalog_input_image)
           embedding_fc = effnet_model_fc(instance_input_image)
        elif config.BACKBONE == 'xception':
           effnet_model_fc = Xception(include_top=False,
                                      input_tensor=None,
                                      input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3),
                                      pooling=None,
                                      post_fix='1')

           effnet_model_wb = Xception(include_top=False,
                                      input_tensor=None,
                                      input_shape=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3),
                                      pooling=None,
                                      post_fix='2')
           embedding_wb = effnet_model_wb(catalog_input_image)
           embedding_fc = effnet_model_fc(instance_input_image)
        embedding_wb = KL.Activation('relu', name='mmid_relu_an')(embedding_wb)
        embedding_fc = KL.Activation('relu', name='mmid_relu_p')(embedding_fc)

        embedding_fc = KL.GlobalAveragePooling2D(data_format='channels_last',
                                             name='MMID_fc')(embedding_fc)

        embedding_wb = KL.GlobalAveragePooling2D(data_format='channels_last',
                                             name='MMID_wb')(embedding_wb)
        inputs = [catalog_input_image, instance_input_image]
        outputs = [embedding_wb, embedding_fc]
        model = KM.Model(inputs, outputs, name='MMID_vec')

        if config.GPU_COUNT > 1:
            from mmid.parallel_model import ParallelModel
            model = ParallelModel(model, config)
        else:
            from mmid.parallel_model import mmid_Npair_loss_graph
            from mmid.parallel_model import top_1_accuracy

            loss = KL.Lambda(lambda x: mmid_Npair_loss_graph(self.config,
                                                             0.02,
                                                             *x), name="MMID_loss")([embedding_fc, embedding_wb])
            acc = KL.Lambda(lambda x: top_1_accuracy(self.config, *x), name="MMID_acc")([embedding_fc, embedding_wb])

            if config.MODE == 'training':
                inputs = [catalog_input_image, instance_input_image]
                outputs = [embedding_wb, embedding_fc, loss, acc]
                model = KM.Model(inputs, outputs, name='MMID_vec')
        return model

    def find_last(self):
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mmid"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        import h5py
        try:
            from keras.engine import saving
        except ImportError:
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
        for layer in layers:
            print("In model: ", layer.name)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        self.set_log_dir(filepath)
    def layers(self):
        return self.keras_model.inner_model.layers if hasattr(self.keras_model, "inner_model")\
            else self.keras_model.layers

    def summary(self):
        print(self.keras_model.summary())

    def get_imagenet_weights(self):

        BASE_WEIGHTS_PATH = (
            'https://github.com/Callidior/keras-applications/'
            'releases/download/efficientnet/')

        WEIGHTS_HASHES = {
            'efficientnet-b0': ('dd631faed10515e2cd08e3b5da0624b3'
                                'f50d523fe69b9b5fdf037365f9f907f0',
                                'e5649d29a9f2dd60380dd05d63389666'
                                '1c36e1f9596e302a305f9ff1774c1bc8'),
            'efficientnet-b1': ('3b88771863db84f3ddea6d722a818719'
                                '04e0fa6288869a0adaa85059094974bb',
                                '5b47361e17c7bd1d21e42add4456960c'
                                '9312f71b57b9f6d548e85b7ad9243bdf'),
            'efficientnet-b2': ('e78c89b8580d907238fd45f8ef200131'
                                '95d198d16135fadc80650b2453f64f6c',
                                'ac3c2de4e43096d2979909dd9ec22119'
                                'c3a34a9fd3cbda9977c1d05f7ebcede9'),
            'efficientnet-b3': ('99725ac825f7ddf5e47c05d333d9fb62'
                                '3faf1640c0b0c7372f855804e1861508',
                                'e70d7ea35fa684f9046e6cc62783940b'
                                'd83d16edc238807fb75c73105d7ffbaa'),
            'efficientnet-b4': ('242890effb990b11fdcc91fceb59cd74'
                                '9388c6b712c96dfb597561d6dae3060a',
                                'eaa6455c773db0f2d4d097f7da771bb7'
                                '25dd8c993ac6f4553b78e12565999fc1'),
            'efficientnet-b5': ('c4cb66916633b7311688dbcf6ed5c35e'
                                '45ce06594181066015c001103998dc67',
                                '14161a20506013aa229abce8fd994b45'
                                'da76b3a29e1c011635376e191c2c2d54')
        }

        from keras.utils.data_utils import get_file
        if self.config.BACKBONE == 'resnet':
            print("loading resnet weights...")
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                     'releases/download/v0.2/'\
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        elif self.config.BACKBONE == 'effnet':
            print("loading effnet weights...")
            model_name = 'efficientnet-b4' 
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        elif self.config.BACKBONE == 'xception':
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                     'releases/download/v0.4/'\
                                     'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='b0042744bf5b25fce3cb969f33bebb97')

        return weights_path

    def compile(self, learning_rate, momentum):
        optimizer = keras.optimizers.Adadelta(
            lr=learning_rate, rho=momentum,
            epsilon=1e-04)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["MMID_loss"]
        acc_names = ["MMID_acc"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))
        for name in acc_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            acc = (
                tf.reduce_mean(layer.output, keepdims=True))
            self.keras_model.metrics_tensors.append(acc)
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mmid\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        self.checkpoint_path = os.path.join(self.log_dir, "mmid_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        assert self.mode == "training", "Create model in training mode."

        layer_regex = {
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "5+": r"(res5.*)|(bn5.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        train_generator = mmid_data_generator(self.config, 
					      batch_size=self.config.BATCH_SIZE, mode='train')

        val_generator = mmid_data_generator(self.config,
                                              batch_size=self.config.BATCH_SIZE, mode='val')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=True),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        if custom_callbacks:
            callbacks += custom_callbacks

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=200,
            workers=workers,
            use_multiprocessing=True,
        )

        self.keras_model.evaluate_generator(
            val_generator,
            steps=20,
            max_queue_size=200,
            workers=workers,
            use_multiprocessing=True)
        self.epoch = max(self.epoch, epochs)

    def mmid_mold_inputs(self, images):
        molded_images = []
        for image in images:

            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)

            molded_images.append(molded_image)
        molded_images = np.stack(molded_images)
        return molded_images

    def mmid_detect(self, images, verbose=0):

        image_shape = images[0].shape
        for g in images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        B, H, W, C = images[0].shape
        wb_image = tf.reshape(images[0], [B, H, W, C])

        B, H, W, C = images[1].shape
        fc_image = tf.reshape(images[1], [B, H, W, C])

        if verbose:
            log("molded_images", images[0])
        embeddings=\
            self.keras_model.predict_on_batch([wb_image, fc_image])
        return embeddings

    def ancestor(self, tensor, name, checked=None):
        checked = checked if checked is not None else []
        if len(checked) > 500:
            return None
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        layers = []
        for l in self.keras_model.layers:
            l = self.find_trainable_layer(l)
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        model = self.keras_model

        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    meta = np.array(
        [image_id] +                  
        list(original_image_shape) +  
        list(image_shape) +           
        list(window) +                
        [scale] +                     
        list(active_class_ids)        
    )
    return meta

def parse_image_meta(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }

def parse_image_meta_graph(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL

def unmold_image(normalized_images, config):
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

def trim_zeros_graph(boxes, name='trim_zeros'):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

def denorm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
EOF
