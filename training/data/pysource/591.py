from keras.models import *
from keras.layers import *
from fpn_network import ResNet, DataGenerator
import tensorflow as tf
from fpn_network import model_utils
import os
import re
import datetime
from fpn_network import utils
from fpn_network.ProposalLayer import ProposalLayer
from fpn_network.DetectionLayer import DetectionLayer
from fpn_network.DetectionTargetLayer import DetectionTargetLayer
from fpn_network import FeaturePyramidNetwork
from fpn_network import Loss_Functions
from fpn_network import RegionProposalNetwork
import keras
import multiprocessing

def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

class FPN():
    def __init__(self, mode, config, model_dir):
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build_model(mode=mode, config=config)

    def build_model(self, mode, config):

        assert mode in ['training', 'inference']

        input_image = Input(shape=(None, None, 3), name="input_image")
        input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        if mode == "training":
            input_rpn_match = Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            input_gt_class_ids = Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            input_gt_boxes = Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            gt_boxes = Lambda(lambda x: model_utils.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
        elif mode == "inference":
            input_anchors = Input(shape=[None, 4], name="input_anchors")

        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = ResNet.resnet_graph(input_image, config.BACKBONE,
                                                    stage5=True, train_bn=config.TRAIN_BN)

        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                    Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                    Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                    Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

        P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        feat_pyr_net_feature_maps = [P2, P3, P4, P5]

        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        rpn = RegionProposalNetwork.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                                    len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = []  
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            active_class_ids = Lambda(
                lambda x: model_utils.parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                   name="input_roi", dtype=np.int32)
                target_rois = Lambda(lambda x: model_utils.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            rois, target_class_ids, target_bbox = \
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes])

            feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox = \
                FeaturePyramidNetwork.fpn_classifier_graph(rois, feat_pyr_net_feature_maps, input_image_meta,
                                                           config.POOL_SIZE, config.NUM_CLASSES,
                                                           train_bn=config.TRAIN_BN,
                                                           fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

            rpn_class_loss = Lambda(lambda x: Loss_Functions.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = Lambda(lambda x: Loss_Functions.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = Lambda(lambda x: Loss_Functions.feat_pyr_net_class_loss_graph(*x),
                                name="feat_pyr_net_class_loss")(
                [target_class_ids, feat_pyr_net_class_logits, active_class_ids])
            bbox_loss = Lambda(lambda x: Loss_Functions.feat_pyr_net_bbox_loss_graph(*x),
                               name="feat_pyr_net_bbox_loss")(
                [target_bbox, target_class_ids, feat_pyr_net_bbox])

            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
            model = Model(inputs, outputs, name='fpn_model')
        else:

            feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox = \
                FeaturePyramidNetwork.fpn_classifier_graph(rpn_rois, feat_pyr_net_feature_maps, input_image_meta,
                                                           config.POOL_SIZE, config.NUM_CLASSES,
                                                           train_bn=config.TRAIN_BN,
                                                           fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            detections = DetectionLayer(config, name="feat_pyr_net_detection")(
                [rpn_rois, feat_pyr_net_class, feat_pyr_net_bbox, input_image_meta])

            model = Model([input_image, input_image_meta, input_anchors],
                          [detections, feat_pyr_net_class, feat_pyr_net_bbox,
                           rpn_rois, rpn_class, rpn_bbox, P2, P3, P4, P5],
                          name='fpn_model')

        return model

    def get_anchors(self, image_shape):
        backbone_shapes = model_utils.compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/fpn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.checkpoint_path = os.path.join(self.log_dir, "fpn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None):
        assert self.mode == "training", "Create model in training mode."

        layer_regex = {
            "heads": r"(feat_pyr_net\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(feat_pyr_net\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(feat_pyr_net\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(feat_pyr_net\_.*)|(rpn\_.*)|(fpn\_.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        print("\n\nBATCH_SIZE: ", self.config.BATCH_SIZE)

        train_generator = DataGenerator.data_generator(train_dataset, self.config, shuffle=True,
                                                       augmentation=augmentation,
                                                       batch_size=self.config.BATCH_SIZE)
        val_generator = DataGenerator.data_generator(val_dataset, self.config, shuffle=True,
                                                     batch_size=self.config.BATCH_SIZE)

        logs_path = self.log_dir + "/training.log"

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=True),
            keras.callbacks.CSVLogger(logs_path, separator=",", append=True),
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
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def compile(self, learning_rate, momentum):
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)

        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "feat_pyr_net_class_loss", "feat_pyr_net_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

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
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        self.set_log_dir(filepath)

    def detect(self, images, verbose=0):
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        detections, fpn_class, fpn_bbox, rpn_rois, rpn_class, rpn_bbox, P2, P3, P4, P5 = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = \
                self.unmold_detections(detections[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            final_P2 = P2[i]
            final_P3 = P3[i]
            final_P4 = P4[i]
            final_P5 = P5[i]
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "P2": final_P2,
                "P3": final_P3,
                "P4": final_P4,
                "P5": final_P5,
            })

        return results

    def mold_inputs(self, images):
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = model_utils.mold_image(molded_image, self.config)
            image_meta = model_utils.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, original_image_shape, image_shape, window):
        print("\n\ndetections: ", detections.shape)

        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        print("\n\nN: ", N)

        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]

        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  
        ww = wx2 - wx1  
        scale = np.array([wh, ww, wh, ww])
        boxes = np.divide(boxes - shift, scale)
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)

        return boxes, class_ids, scores
EOF
