from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.layers import Input, Flatten
from keras.constraints import max_norm
import tensorflow as tf
from keras import regularizers


def ECoGNet(settings, nb_classes, Chans, Samples, num_input, norm_rate=0.25):
    """
    Constructs the ECoGNet model.

    Inputs:
      nb_classes      : Number of classes to classify
      Chans, Samples  : Number of channels and time points in the input neural data
      num_input       : Number of input channels
      norm_rate       : Maximum norm for dense layer weights
    """

    # Determine the type of dropout
    if settings['hyper_param']['dropoutType'] == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif settings['hyper_param']['dropoutType'] == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # Initialize lists to hold inputs
    input_all = []
    block11 = []

    # Create input layers and first dense layers for each input channel
    for i in range(num_input):
        input_all.append(Input(shape=(1, Samples, Chans[i])))
        block11.append(tf.transpose(Dense(units=settings['n_ROI'],
                                          kernel_constraint=max_norm(norm_rate),
                                          kernel_regularizer=regularizers.L2(settings['coef_reg']))(input_all[i]),
                                    (0, 1, 3, 2)))

    # Sum the dense layers' outputs along the first axis
    block22 = tf.reduce_sum(block11, axis=0)

    block1 = Conv2D(settings['hyper_param']['F1'], (1, settings['hyper_param']['kernLength']), padding='same',
                    input_shape=(num_input, settings['n_ROI'], Samples),
                    use_bias=False,
                    data_format="channels_first")(block22)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((settings['n_ROI'], 1), use_bias=False,
                             depth_multiplier=settings['hyper_param']['D'],
                             depthwise_constraint=max_norm(1.),
                             data_format="channels_first")(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4), data_format="channels_first")(block1)
    block1 = dropoutType(settings['hyper_param']['dropoutRate'])(block1)

    block2 = SeparableConv2D(settings['hyper_param']['F1'] * settings['hyper_param']['D'],
                             (1, settings['hyper_param']['kernLength_sep']),
                             use_bias=False, padding='same', data_format="channels_first")(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format="channels_first")(block2)
    block2 = dropoutType(settings['hyper_param']['dropoutRate'])(block2)

    flatten = Flatten()(block2)

    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_all, outputs=softmax)
