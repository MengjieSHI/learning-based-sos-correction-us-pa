import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import *
from keras.initializers import GlorotUniform
from keras.regularizers import l2
from keras import Model



def encoder_block(input_shape=None, dropout=0.5):
    '''
    :param input_shape: (None, channel number, time series length, frame) e.g. (None, 128, 1024, 1)
    :param dropout: dropout rate
    :return: FCN encoder part (keras model)
    '''
    # Input
    rf_input = Input(shape=input_shape)
    # add Gaussain noise with stddev 1
    noisy_rf_input = GaussianNoise(1)(rf_input)

    # block 1
    x = Conv2D(32, (3, 15), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='encoder_conv1')(noisy_rf_input)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 13), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='encoder_conv2')(x)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 11), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='encoder_conv3')(x)
    x = BatchNormalization()(x, training=True)
    x = Conv2D(32, (3, 9), (1, 2), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='encoder_conv4')(x)
    x = Dropout(dropout, name='dropout4')(x)
    x = BatchNormalization()(x, training=True)

    # block 2
    x = Conv2D(64, (3, 7), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='encoder_conv5')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='encoder_pool1')(x)
    x = Dropout(dropout, name='dropout5')(x)
    x = BatchNormalization()(x, training=True)

    # block 3
    x = Conv2D(128, (3, 5), (1, 1), padding='same', activation='LeakyReLU',kernel_initializer=GlorotUniform(), name='encoder_conv6')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='encoder_pool2')(x)
    x = BatchNormalization()(x, training=True)

    # block 4
    x = Conv2D(256, (3, 3), (1, 1), padding='same', activation='LeakyReLU',kernel_initializer=GlorotUniform(), name='encoder_conv7')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='encoder_pool3')(x)
    x = Dropout(dropout, name='dropout7')(x)
    x = BatchNormalization()(x, training=True)

    model = Model(rf_input, x)
    model.summary()

    return model


def decoder_block(encoder_block):
    '''

    :param encoder_block: FCN encoder part
    :return: FCN (keras model)
    '''
    # block 1
    x = Conv2D(128, (3, 3), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='decoder_conv1')(encoder_block.get_layer('encoder_pool3').output)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer('encoder_pool2').output, x])

    # block 2
    x = Conv2D(64, (3, 5), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='decoder_conv2')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer('encoder_pool1').output, x])

    # block 3
    x = Conv2D(32, (3, 7), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(), name='decoder_conv3')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)
    x = Concatenate()([encoder_block.get_layer('encoder_conv4').output, x])

    # block 4
    x = Conv2D(32, (3, 9), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(),
               name='decoder_conv4')(x)
    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = BatchNormalization()(x, training=True)

    # block 5
    x = Conv2D(32, (3, 11), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(),
               name='decoder_conv5')(x)
    x = Resizing(384, 384)(x)

    # block 6
    x = Conv2D(32, (3, 3), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(),
               name='decoder_conv6')(x)
    x = BatchNormalization()(x, training=True)

    # 1*1 conv
    x = Conv2D(1, (1, 1), (1, 1), padding='same', activation='LeakyReLU', kernel_initializer=GlorotUniform(),
               name='decoder_conv7')(x)

    model = Model(encoder_block.input, x)
    return model

def TF_block(base_model, input_shape=None):
    '''

    :param base_model: pre-train model using pure in silico
    :param input_shape: (None, channel number, time series length, frame) e.g. (None, 128, 1024, 1)
    :return: results with finetuned by TF
    '''
    rf_input = Input(shape=input_shape)
    x_initial = base_model(rf_input, training=False)
    x = Conv2D(1, (3, 3), (1, 1), padding='same', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01))(x_initial)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, (3, 3), (1, 1), padding='same', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    x = Add()([x_initial, x])
    x = LeakyReLU()(x)

    model = Model(rf_input, x)
    return model










