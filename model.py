from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, concatenate
from keras.layers import AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

from config import *
import os

os.environ["CUDA_INVISIBLE_DEVICES"] = "1"


def GoogLeNet(input_shape=(224, 224, 3)):
    # stem
    input_layer = Input(shape=input_shape)
    x = conv2d_block(input_layer,
                     filters=64,
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     padding="same",
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     name="conv_1_7x7/2",
                     pooling=True
                     )
    x = conv2d_block(x,
                     filters=64,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding="same",
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     name="conv_2a_1x1/1",
                     pooling=False
                     )
    x = conv2d_block(x,
                     filters=192,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="same",
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     name="conv_2b_3x3/1",
                     pooling=True
                     )

    # Inception 3a
    x = inception_block(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name="inception_block_3a")
    # inception 3b
    x = inception_block(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name="inception_block_3b")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same", )(x)

    # inception 4a
    x = inception_block(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name="inception_block_4a")

    # auxiliary output 1
    x1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same", )(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, )(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation="softmax", name="auxiliary_output_1")(x1)

    # inception 4b
    x = inception_block(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name="inception_block_4b")
    # inception 4c
    x = inception_block(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name="inception_block_4c")
    # inception 4d
    x = inception_block(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name="inception_block_4d")

    # auxiliary output 2
    x2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    x2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same", )(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, )(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation="softmax", name="auxiliary_output_2")(x2)

    # inception 4e
    x = inception_block(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name="inception_block_4e")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="max_pool_4_3x3/2")(x)

    # inception 5a
    x = inception_block(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name="inception_block_5a")
    # inception 5b
    x = inception_block(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name="inception_block_5b")

    x = GlobalAveragePooling2D(name="avg_pool_5_3x3/1")(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(input_layer, [x, x1, x2], name="inception_v1")
    model.summary()
    return model


def conv2d_block(x, filters,
                 kernel_size,
                 strides,
                 padding,
                 kernel_initializer,
                 bias_initializer,
                 name,
                 pooling=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               name=name)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if pooling:
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    return x


def inception_block(x,
                    filters_1x1,
                    filters_3x3_reduce,
                    filters_3x3,
                    filters_5x5_reduce,
                    filters_5x5,
                    filters_pool_proj,
                    name=None):
    conv_1x1 = Conv2D(filters_1x1,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      activation="relu",
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      activation="relu",
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      activation="relu",
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      activation="relu",
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5,
                      kernel_size=(5, 5),
                      strides=(1, 1),
                      padding="same",
                      activation="relu",
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(filters_pool_proj,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding="same",
                       activation="relu",
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


if __name__ == "__main__":
    GoogLeNet()
