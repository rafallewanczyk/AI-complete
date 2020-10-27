from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D
from tensorflow.keras.models import Model

from notekeras.component import resnet_bolck as B
from notekeras.layer import BatchNormalizationFreeze

class ResNet1D(Model):

    def __init__(self, inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None,
                 *args, **kwargs):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = layers.ZeroPadding1D(padding=3, name="padding_conv1")(inputs)
        x = layers.Conv1D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
        x = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = layers.Activation("relu", name="conv1_relu")(x)
        x = layers.MaxPooling1D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling1D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet1D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            super(ResNet1D, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

class ResNet1D18(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(ResNet1D18, self).__init__(inputs, blocks, block=B.basic_1d, include_top=include_top, classes=classes,
                                         freeze_bn=freeze_bn, *args, **kwargs)

class ResNet1D34(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(ResNet1D34, self).__init__(inputs, blocks, block=B.basic_1d, include_top=include_top, classes=classes,
                                         freeze_bn=freeze_bn, *args, **kwargs)

class ResNet1D50(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet1D50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet1D101(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet1D152(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet1D200(ResNet1D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet1D200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_1d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet2D(Model):

    def __init__(self, inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None,
                 *args, **kwargs):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)
        x = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = Activation("relu", name="conv1_relu")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]),
                          freeze_bn=freeze_bn)(x)

            features *= 2
            outputs.append(x)

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet2D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            super(ResNet2D, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

class ResNet2D18(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(ResNet2D18, self).__init__(
            inputs,
            blocks,
            block=B.basic_2d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet2D34(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(ResNet2D34, self).__init__(
            inputs,
            blocks,
            block=B.basic_2d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet2D50(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet2D50, self).__init__(inputs, blocks,
                                         numerical_names=numerical_names,
                                         block=B.bottleneck_2d,
                                         include_top=include_top,
                                         classes=classes,
                                         freeze_bn=freeze_bn,
                                         *args,
                                         **kwargs)

class ResNet2D101(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet2D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet2D152(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet2D152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet2D200(ResNet2D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet2D200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D(Model):

    def __init__(
            self,
            inputs,
            blocks,
            block,
            include_top=True,
            classes=1000,
            freeze_bn=True,
            numerical_names=None,
            *args,
            **kwargs
    ):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = keras.layers.ZeroPadding3D(padding=3, name="padding_conv1")(inputs)
        x = keras.layers.Conv3D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
        x = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling3D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling3D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet3D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            super(ResNet3D, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

class ResNet3D18(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(ResNet3D18, self).__init__(
            inputs,
            blocks,
            block=B.basic_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D34(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(ResNet3D34, self).__init__(
            inputs,
            blocks,
            block=B.basic_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D50(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet3D50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D101(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(ResNet3D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D152(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet3D152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class ResNet3D200(ResNet3D):

    def __init__(self, inputs, blocks=None, include_top=True, classes=1000, freeze_bn=False, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(ResNet3D200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_3d,
            include_top=include_top,
            classes=classes,
            freeze_bn=freeze_bn,
            *args,
            **kwargs
        )

class FPN2D(Model):
    def __init__(
            self,
            inputs,
            blocks,
            block,
            freeze_bn=True,
            numerical_names=None,
            *args,
            **kwargs
    ):
        if keras.backend.image_data_format() == "channels_last":
            axis = 3
        else:
            axis = 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)
        x = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn
                )(x)

            features *= 2

            outputs.append(x)

        c2, c3, c4, c5 = outputs

        pyramid_5 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c5_reduced"
        )(c5)

        upsampled_p5 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p5_upsampled",
            size=(2, 2)
        )(pyramid_5)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c4_reduced"
        )(c4)

        pyramid_4 = keras.layers.Add(
            name="p4_merged"
        )([upsampled_p5, pyramid_4])

        upsampled_p4 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p4_upsampled",
            size=(2, 2)
        )(pyramid_4)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p4"
        )(pyramid_4)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c3_reduced"
        )(c3)

        pyramid_3 = keras.layers.Add(
            name="p3_merged"
        )([upsampled_p4, pyramid_3])

        upsampled_p3 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p3_upsampled",
            size=(2, 2)
        )(pyramid_3)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p3"
        )(pyramid_3)

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c2_reduced"
        )(c2)

        pyramid_2 = keras.layers.Add(
            name="p2_merged"
        )([upsampled_p3, pyramid_2])

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p2"
        )(pyramid_2)

        pyramid_6 = keras.layers.MaxPooling2D(strides=2, name="p6")(pyramid_5)

        outputs = [
            pyramid_2,
            pyramid_3,
            pyramid_4,
            pyramid_5,
            pyramid_6
        ]

        super(FPN2D, self).__init__(
            inputs=inputs,
            outputs=outputs,
            *args,
            **kwargs
        )

class FPN2D50(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(FPN2D50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            *args,
            **kwargs
        )

class FPN2D18(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [2, 2, 2, 2]

        super(FPN2D18, self).__init__(
            inputs,
            blocks,
            block=B.basic_2d,
            *args,
            **kwargs
        )

class FPN2D34(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 6, 3]

        super(FPN2D34, self).__init__(
            inputs,
            blocks,
            block=B.basic_2d,
            *args,
            **kwargs
        )

class FPN2D101(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 4, 23, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D101, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            *args,
            **kwargs
        )

class FPN2D152(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 8, 36, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D152, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            *args,
            **kwargs
        )

class FPN2D200(FPN2D):
    def __init__(self, inputs, blocks=None, *args, **kwargs):
        if blocks is None:
            blocks = [3, 24, 36, 3]

        numerical_names = [False, True, True, False]

        super(FPN2D200, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=B.bottleneck_2d,
            *args,
            **kwargs
        )

def TimeDistributedResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, *args, **kwargs):
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    x = keras.layers.TimeDistributed(keras.layers.ZeroPadding2D(padding=3), name="padding_conv1")(inputs)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False), name="conv1")(x)
    x = keras.layers.TimeDistributed(BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn),
                                     name="bn_conv1")(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation("relu"), name="conv1_relu")(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"), name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(blocks[stage_id] > 6), freeze_bn=freeze_bn)(x)

        features *= 2
        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D(), name="pool5")(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"), name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)

def TimeDistributedResNet18(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [2, 2, 2, 2]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_basic_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)

def TimeDistributedResNet34(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_basic_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)

def TimeDistributedResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_bottleneck_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)

def TimeDistributedResNet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 4, 23, 3]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_bottleneck_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)

def TimeDistributedResNet152(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 8, 36, 3]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_bottleneck_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)

def TimeDistributedResNet200(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 24, 36, 3]

    return TimeDistributedResNet(inputs, blocks, block=B.time_distributed_bottleneck_2d,
                                 include_top=include_top, classes=classes, *args, **kwargs)
EOF
