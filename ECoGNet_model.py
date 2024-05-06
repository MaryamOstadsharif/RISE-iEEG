from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
import tensorflow as tf
from keras import regularizers
import numpy as np


def ECoGNet(nb_classes, Chans, Samples, dropoutRate, kernLength, F1, n_ROI, D, F2, dropoutType, kernLength_sep,
            num_input, use_transfer, num_input_pretrained_model, coef_reg, norm_rate=0.25):
    """
    Inputs:

      nb_classes      : Number of classes to classify
      Chans, Samples  : Number of channels and time points in the input neural data
      dropoutRate     : Dropout fraction
      kernLength      : Length of temporal convolution kernel in first layer
      F1, F2          : Number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn; we used F2 = F1 * D, same as EEGNet paper
      D               : Number of spatial filters to learn within each temporal
                        convolution
      norm_rate       : Maximum norm for dense layer weights
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string
      ROIs            : Number common brain regions projecting to (only used if projectROIs == True)
      kernLength_sep  : Length of temporal convolution kernel in separable convolution layer
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input_all = []
    block11 = []
    for i in range(num_input):
        input_all.append(Input(shape=(1, Samples, Chans[i])))
        block11.append(tf.transpose(Dense(units=n_ROI,
                                          kernel_constraint=max_norm(norm_rate),
                                          kernel_regularizer=regularizers.L2(coef_reg))(input_all[i]), (0, 1, 3, 2)))

    # block22 = tf.concat(block11, axis=1)
    block22 = tf.reduce_sum(block11, axis=0)

    ##################################################################
    if use_transfer:
        pad_width = ((0, 0), (0, num_input_pretrained_model - num_input), (0, 0), (0, 0))
        block22 = tf.pad(block22, pad_width, 'CONSTANT')

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(num_input, n_ROI, Samples),
                    use_bias=False,
                    data_format="channels_first")(block22)

    block1 = BatchNormalization(axis=1)(block1)

    # Depthwise kernel acts over all electrodes or brain regions

    block1 = DepthwiseConv2D((n_ROI, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.),
                             data_format="channels_first")(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4), data_format="channels_first")(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, kernLength_sep),
                             use_bias=False, padding='same', data_format="channels_first")(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format="channels_first")(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten()(block2)

    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_all, outputs=softmax)
