
import argparse
import os

def ensure_exist_save_dir(path):
    dirpath = os.path.dirname(path)
    abs_dirpath = os.path.abspath(dirpath)
    os.makedirs(abs_dirpath, exist_ok=True)

def generate_common_model(save_model_path, save_image_path=None):  

    from keras.utils import plot_model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.convolutional import Conv2D
    from keras.layers.core import Activation, Dropout, Flatten
    from keras.layers.pooling import MaxPool2D

    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(1.0))

    model.add(Dense(1, activation='sigmoid'))
    json_model = model.to_json()

    ensure_exist_save_dir(save_model_path)
    with open(save_model_path, "w") as f:
        f.write(json_model)

    if save_image_path is not None:
        ensure_exist_save_dir(save_image_path)
        plot_model(model,
                   to_file=save_image_path,
                   show_shapes=True)

    return model

def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    if not isinstance(layer_id, list):
        layer_id = [layer_id, ]
    if not isinstance(new_layer, list):
        new_layer = [new_layer, ]

    layers = [l for l in model.layers]

    model_input = None
    if 0 in layer_id:
        x = new_layer[0]  
        model_input = x
    else:
        x = layers[0].output
        model_input = layers[0].input
    for i in range(1, len(layers)):
        if i in layer_id:
            idx = layer_id.index(i)
            x = new_layer[idx](x)
        else:
            x = layers[i](x)

    new_model = Model(inputs=model_input, outputs=x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    if not isinstance(layer_id, list):
        layer_id = [layer_id, ]
    if not isinstance(new_layer, list):
        new_layer = [new_layer, ]

    layers = [l for l in model.layers]

    model_input = None
    x = None
    while 0 in layer_id:
        if x is None:
            x = new_layer[0]  
            model_input = new_layer[0]
        else:
            x = new_layer[0](x)
        idx = layer_id.index(0)
        layer_id.remove(0)
        del new_layer[idx]
    if x is None:
        model_input = layers[0].input
        x = layers[0].output
    x = layers[0](x)
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(inputs=model_input, outputs=x)
    return new_model

def create_bench_model(inputs):
    from keras.models import Model
    from keras.layers import Input, Conv3D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Dense

    x = Conv3D(4,(3,3,3),padding = "SAME",activation= "relu")(inputs)
    x = BatchNormalization(axis=3, epsilon=1.001e-5,
                                    name='conv1_bn')(x)

    x = Conv3D(4,(3,3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization(axis=3, epsilon=1.001e-5,
                                    name='conv2_bn')(x)

    x = Conv3D(1,(3,3,3),padding = "SAME",activation= "relu")(x)

    return Model(input = inputs, output = x)

def generate_model_base(preset, width, height, channel, weights_init):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D
    from keras.models import Model

    input_tensor = Input(shape=(width, height, channel))
    conv_base = None
    prediction_layer = None

    if preset.upper() == "bench".upper():
        conv_base = create_bench_model(input_tensor)
        prediction_layer = conv_base
    elif preset.upper() == "VGG16".upper():
        from keras.applications import VGG16
        conv_base = None
        if channel == 3:
            conv_base = VGG16(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, channel)
                              )
        else:
            conv_base = VGG16(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, 3)
                              )

            conv_base.layers.pop(0)
            conv_base.layers.pop(0)
            input_layer = Input(shape=(width, height, channel), name='multi_input')
            block1_conv1_new = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='glorot_uniform',
                                      name='block1_conv1_new')

            conv_base = insert_intermediate_layer_in_keras(conv_base, [0, 0], [input_layer, block1_conv1_new])

        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.get_output_at(-1)  
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.2, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "VGG19".upper():
        from keras.applications import VGG19
        conv_base = None
        if channel == 3:
            conv_base = VGG19(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, channel)
                              )
        else:
            conv_base = VGG19(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, 3)
                              )

            conv_base.layers.pop(0)
            conv_base.layers.pop(0)
            input_layer = Input(shape=(width, height, channel), name='multi_input')
            block1_conv1_new = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='glorot_uniform',
                                      name='block1_conv1_new')

            conv_base = insert_intermediate_layer_in_keras(conv_base, [0, 0], [input_layer, block1_conv1_new])

        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.get_output_at(-1)  
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.2, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "VGG16BN".upper():
        from model import VGG16BN
        conv_base = None
        if channel == 3:
            conv_base = VGG16BN(weights=weights_init,
                                include_top=False,
                                pooling='avg',
                                kernel_initializer='glorot_uniform',
                                input_shape=(width, height, channel)
                                )
        else:
            conv_base = VGG16BN(weights=weights_init,
                                include_top=False,
                                pooling='avg',
                                kernel_initializer='glorot_uniform',
                                input_shape=(width, height, 3)
                                )

            conv_base.layers.pop(0)
            conv_base.layers.pop(0)
            input_layer = Input(shape=(width, height, channel), name='multi_input')
            block1_conv1_new = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='glorot_uniform',
                                      name='block1_conv1_new')

            conv_base = insert_intermediate_layer_in_keras(conv_base, [0, 0], [input_layer, block1_conv1_new])

        output_layer = conv_base.layers[-1]
        x = output_layer.get_output_at(-1)  
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.2, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "VGG19BN".upper():
        from model import VGG19BN
        conv_base = None
        if channel == 3:
            conv_base = VGG19BN(weights=weights_init,
                                include_top=False,
                                pooling='avg',
                                kernel_initializer='glorot_uniform',
                                input_shape=(width, height, channel)
                                )
        else:
            conv_base = VGG19BN(weights=weights_init,
                                include_top=False,
                                pooling='avg',
                                kernel_initializer='glorot_uniform',
                                input_shape=(width, height, 3)
                                )

            conv_base.layers.pop(0)
            conv_base.layers.pop(0)
            input_layer = Input(shape=(width, height, channel), name='multi_input')
            block1_conv1_new = Conv2D(64, (3, 3),
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='glorot_uniform',
                                      name='block1_conv1_new')

            conv_base = insert_intermediate_layer_in_keras(conv_base, [0, 0], [input_layer, block1_conv1_new])

        output_layer = conv_base.layers[-1]
        x = output_layer.get_output_at(-1)  
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.2, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet20".upper():
        from model.resnet import ResNet20
        conv_base = ResNet20(weights=weights_init,
                             include_top=True,
                             input_shape=(width, height, channel),
                             input_tensor=input_tensor
                             )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet50".upper():
        from model.resnet import ResNet50
        conv_base = ResNet50(weights=weights_init,
                             include_top=True,
                             input_shape=(width, height, channel),
                             input_tensor=input_tensor
                             )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet101".upper():
        from model.resnet import ResNet101
        conv_base = ResNet101(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, channel),
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet152".upper():
        from model.resnet import ResNet152
        conv_base = ResNet152(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, channel),
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet50V2".upper():
        from model.resnet_v2 import ResNet50V2
        conv_base = ResNet50V2(weights=weights_init,
                               include_top=True,
                               input_shape=(width, height, channel),
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet101V2".upper():
        from model.resnet_v2 import ResNet101V2
        conv_base = ResNet101V2(weights=weights_init,
                                include_top=True,
                                input_shape=(width, height, channel),
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNet152V2".upper():
        from model.resnet_v2 import ResNet152V2
        conv_base = ResNet152V2(weights=weights_init,
                                include_top=True,
                                input_shape=(width, height, channel),
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNeXt50".upper():
        from model.resnext import ResNeXt50
        conv_base = ResNeXt50(weights=weights_init,
                              include_top=True,
                              input_shape=(width, height, channel),
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "ResNeXt101".upper():
        from model.resnext import ResNeXt101
        conv_base = ResNeXt101(weights=weights_init,
                               include_top=True,
                               input_shape=(width, height, channel),
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        x = Dropout(0.5, name='fc_dropout')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "InceptionV3".upper():
        from keras.applications import InceptionV3
        conv_base = InceptionV3(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "InceptionResNetV2".upper():
        from keras.applications import InceptionResNetV2
        conv_base = InceptionResNetV2(weights=weights_init,
                                      include_top=True,
                                      input_tensor=input_tensor
                                      )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "DenseNet121".upper():
        from keras.applications import DenseNet121
        conv_base = DenseNet121(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "DenseNet169".upper():
        from keras.applications import DenseNet169
        conv_base = DenseNet169(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "DenseNet201".upper():
        from keras.applications import DenseNet201
        conv_base = DenseNet201(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "Xception".upper():
        from keras.applications import Xception
        conv_base = Xception(weights=weights_init,
                             include_top=True,
                             input_tensor=input_tensor
                             )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEDenseNetImageNet121".upper():
        from model import SEDenseNetImageNet121
        conv_base = SEDenseNetImageNet121(weights=weights_init,
                                          include_top=True,
                                          input_tensor=input_tensor
                                          )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEDenseNetImageNet169".upper():
        from model import SEDenseNetImageNet169
        conv_base = SEDenseNetImageNet169(weights=weights_init,
                                          include_top=True,
                                          input_tensor=input_tensor
                                          )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEDenseNetImageNet201".upper():
        from model import SEDenseNetImageNet201
        conv_base = SEDenseNetImageNet201(weights=weights_init,
                                          include_top=True,
                                          input_tensor=input_tensor
                                          )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEDenseNetImageNet264".upper():
        from model import SEDenseNetImageNet264
        conv_base = SEDenseNetImageNet264(weights=weights_init,
                                          include_top=True,
                                          input_tensor=input_tensor
                                          )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEDenseNetImageNet161".upper():
        from model import SEDenseNetImageNet161
        conv_base = SEDenseNetImageNet161(weights=weights_init,
                                          include_top=True,
                                          input_tensor=input_tensor
                                          )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEInceptionResNetV2".upper():
        from model import SEInceptionResNetV2
        conv_base = SEInceptionResNetV2(weights=weights_init,
                                        include_top=True,
                                        input_tensor=input_tensor
                                        )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEInceptionV3".upper():
        from model import SEInceptionV3
        conv_base = SEInceptionV3(weights=weights_init,
                                  include_top=True,
                                  input_tensor=input_tensor
                                  )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEMobileNet".upper():
        from model import SEMobileNet
        conv_base = SEMobileNet(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet6".upper():
        from model import SEResNet6
        conv_base = SEResNet6(weights=weights_init,
                              include_top=True,
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet8".upper():
        from model import SEResNet8
        conv_base = SEResNet8(weights=weights_init,
                              include_top=True,
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet10".upper():
        from model import SEResNet10
        conv_base = SEResNet10(weights=weights_init,
                               include_top=True,
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet18".upper():
        from model import SEResNet18
        conv_base = SEResNet18(weights=weights_init,
                               include_top=True,
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet34".upper():
        from model import SEResNet34
        conv_base = SEResNet34(weights=weights_init,
                               include_top=True,
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet50".upper():
        from model import SEResNet50
        conv_base = SEResNet50(weights=weights_init,
                               include_top=True,
                               input_tensor=input_tensor
                               )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet101".upper():
        from model import SEResNet101
        conv_base = SEResNet101(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNet154".upper():
        from model import SEResNet154
        conv_base = SEResNet154(weights=weights_init,
                                include_top=True,
                                input_tensor=input_tensor
                                )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    elif preset.upper() == "SEResNext".upper():
        from model import SEResNext
        conv_base = SEResNext(weights=weights_init,
                              include_top=True,
                              input_tensor=input_tensor
                              )
        conv_base.layers.pop()
        output_layer = conv_base.layers[-1]
        x = output_layer.output
        x = BatchNormalization(name='fc_bachnorm')(x)
        prediction_layer = Dense(class_num, activation='softmax',
                                 kernel_initializer='glorot_uniform', name='prediction')(x)

    else:
        raise ValueError('unknown model name : {}'.format(preset))

    model = Model(inputs=conv_base.input,
                  outputs=prediction_layer, name='classification_model')
    return model

def model_tuning(model_base):
    from keras.models import Sequential
    model = Sequential()
    model.add(model_base)

    return model

def save_network(model, path):
    ensure_exist_save_dir(path)
    json_model = model.to_json()
    with open(path, "w") as f:
        f.write(json_model)

def save_graph(model, path):
    from keras.utils import plot_model
    ensure_exist_save_dir(path)
    plot_model(model,
               to_file=path,
               show_shapes=True)

def list_preset(output=True):
    preset = [
        'VGG16',
        'VGG19',
        'ResNet50',
        'ResNet101',
        'ResNet152',
        'ResNet50V2',
        'ResNet101V2',
        'ResNet152V2',
        'ResNeXt50',
        'ResNeXt101',
        'InceptionV3',
        'InceptionResNetV2',
        'DenseNet121',
        'DenseNet169',
        'DenseNet201',
        'VGG16BN',
        'VGG19BN',
        'Xception',
        'SEDenseNetImageNet121',
        'SEDenseNetImageNet169',
        'SEDenseNetImageNet201',
        'SEDenseNetImageNet264',
        'SEDenseNetImageNet161',
        'SEInceptionResNetV2',
        'SEInceptionV3',
        'SEMobileNet',
        'SEResNet8',
        'SEResNet10',
        'SEResNet18',
        'SEResNet34',
        'SEResNet50',
        'SEResNet101',
        'SEResNet154',
        'SEResNext',
    ]
    if output:
        print('Keras preset:')
        for p in preset:
            print('  {}'.format(p))
    return preset

def get_shape_in(shape_in):
    words = shape_in.split(',')
    if len(words) != 3:
        raise ValueError('shape_in {} is wrong value.'.format(v))

    return [int(v) for v in words]

def print_input(args):
    for arg in vars(args):
        print('{} : {}'.format(arg, getattr(args, arg)))

def info(args):
    if args.verbose:
        print_input(args)

    if args.in_network is None or not os.path.exists(args.in_network):
        raise ValueError('File was not found. {}'.format(args.in_network))

    path = args.in_network

    _, ext = os.path.splitext(path)
    with open(path, mode='r', encoding='utf-8') as f:
        content = f.read()
        if ext == '.json':
            from keras.models import model_from_json
            model = model_from_json(content)
        if ext == '.yaml':
            from keras.models import model_from_yaml
            model = model_from_yaml(content)

    if args.verbose:
        model.summary()

def create(args):
    if args.verbose:
        print_input(args)

    from keras.backend import tensorflow_backend as backend

    width, height, channel = get_shape_in(args.shape_in)
    if args.verbose:
        print("preset: {}".format(args.preset))
        print("width: {}, height: {}, channel: {}".format(width, height, channel))
        print("out : {}".format(args.shape_out))
        print("init: {}".format(args.init))
    model = generate_model_base(preset=args.preset,
                                width=width,
                                height=height,
                                channel=channel,
                                class_num=int(args.shape_out),
                                weights_init=args.init
                                )

    from keras.optimizers import Adagrad
    model.compile(loss='categorical_crossentropy', optimizer=Adagrad(), metrics=['accuracy'])
    save_network(model=model, path=args.out_network)
    if args.out_image is not None:
        save_graph(model=model, path=args.out_image)
        print("save_graph:{}".format(args.out_image))

    if args.out_library is None:
        raise ValueError('no parameter out-library : {}'.format(args.out_library))
    else:
        ensure_exist_save_dir(os.path.dirname(args.out_library))
        model.save_weights(args.out_library)

    if args.verbose:
        model.summary()

    backend.clear_session()
    print('process complete!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_argument_parser(parser)
EOF
