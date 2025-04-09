import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, Activation, Dropout, Flatten, Conv2D, AveragePooling2D,
    SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D
)
from keras.constraints import max_norm
from keras import regularizers


def RISEiEEG(settings, nb_classes, Chans, Samples, num_input, norm_rate=0.25):
    """
    Constructs the RISE-iEEG model architecture.

    Args:
        settings    : Custom settings object containing hyperparameters.
        nb_classes  : Number of output classes.
        Chans       : List of channel counts for each input.
        Samples     : Number of time samples per input.
        num_input   : Number of individual input streams (patients/channels).
        norm_rate   : Max norm constraint for weights.

    Returns:
        Compiled Keras model.
    """

    # Select dropout layer type
    dropout_type = settings.hyper_param.get('dropoutType', 'Dropout')
    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError("dropoutType must be 'SpatialDropout2D' or 'Dropout'.")

    input_all = []
    dense_transformed = []

    # Apply a Dense layer to each input, transpose to expected format
    for i in range(num_input):
        input_layer = Input(shape=(1, Samples, Chans[i]))
        input_all.append(input_layer)

        dense_out = Dense(
            units=settings.n_ROI,
            kernel_constraint=max_norm(norm_rate),
            kernel_regularizer=regularizers.L2(settings.coef_reg)
        )(input_layer)

        # Transpose to (batch, 1, n_ROI, time)
        transposed = tf.transpose(dense_out, perm=(0, 1, 3, 2))
        dense_transformed.append(transposed)

    # Merge all inputs by summing along the input axis
    merged_input = tf.reduce_sum(dense_transformed, axis=0)

    # First Convolution Block
    block1 = Conv2D(
        filters=settings.hyper_param['F1'],
        kernel_size=(1, settings.hyper_param['kernLength']),
        padding='same',
        use_bias=False,
        data_format="channels_first"
    )(merged_input)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D(
        kernel_size=(settings.n_ROI, 1),
        use_bias=False,
        depth_multiplier=settings.hyper_param['D'],
        depthwise_constraint=max_norm(1.),
        data_format="channels_first"
    )(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size=(1, 4), data_format="channels_first")(block1)
    block1 = DropoutLayer(settings.hyper_param['dropoutRate'])(block1)

    # Second Convolution Block
    block2 = SeparableConv2D(
        filters=settings.hyper_param['F1'] * settings.hyper_param['D'],
        kernel_size=(1, settings.hyper_param['kernLength_sep']),
        padding='same',
        use_bias=False,
        data_format="channels_first"
    )(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size=(1, 8), data_format="channels_first")(block2)
    block2 = DropoutLayer(settings.hyper_param['dropoutRate'])(block2)

    # Classification Head
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    output = Activation('softmax')(dense)

    return Model(inputs=input_all, outputs=output)
