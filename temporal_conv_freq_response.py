import numpy as np
from scipy.fftpack import fft
from scipy.signal import hann
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from keras import backend as K
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU

weighted_pow_medians_all = []
for curr_fold in range(5):
    # loadname = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-03-06-12-16-57/' \
    #            'accuracy/checkpoint_gen__fold' + str(curr_fold) + '.h5'
    loadname = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-02-14-23-34-37/accuracy/' \
               'checkpoint_gen__fold' + str(curr_fold) + '.h5'
    model_curr = tf.keras.models.load_model(loadname)

    temp_conv_layer = 'conv2d'
    output_layer = temp_conv_layer
    srate_new = 100
    # tLen = int(model_curr.get_layer('conv2d').input.get_shape()[-1])
    # chLen = int(model_curr.get_layer('conv2d').input.get_shape()[-2])
    # input1 = Input(shape=model_curr.get_layer(output_layer).input_shape[1:])
    # intermediate_layer_model = Model(inputs=input1,outputs=model_curr.get_layer(output_layer).output)
    input1 = Input(shape=model_curr.get_layer('conv2d').input_shape[1:])
    # block1 = Conv2D(2, (1, 20), padding='same',
    #                 input_shape=(29, 55, 200),
    #                 use_bias=False,
    #                 data_format="channels_first")(input1)
    block1 = Conv2D(5, (1, 60), padding='same',
                    input_shape=(12, 20, 200),
                    use_bias=False,
                    data_format="channels_first")(input1)
    intermediate_layer_model = Model(inputs=input1,
                                     outputs=block1)
    intermediate_layer_model.layers[-1].set_weights(model_curr.layers[-15].get_weights())

    nrows = int(intermediate_layer_model.input[0].shape[-2])
    ncols = int(intermediate_layer_model.input[0].shape[-1])
    # data_in = np.zeros([1, 29, nrows, ncols])
    data_in = np.zeros([1, 12, nrows, ncols])

    for i in range(nrows):
        data_in[..., i, :] = np.random.standard_normal(ncols)

    filt_dat = intermediate_layer_model.predict(data_in)  # error pops up if running on CPU

    # Compute spectral power density for orignal and filtered dummy data
    w = hann(data_in.shape[-1])
    pow_orig = np.mean(fft(data_in * w), axis=1)
    pow_filt = fft(filt_dat * w)
    f = np.fft.fftfreq(pow_orig.shape[-1], d=1 / srate_new)

    pow_orig = pow_orig[..., f > 0]
    pow_filt = pow_filt[..., f > 0]
    f = f[f > 0]

    pow_diff = np.zeros_like(pow_filt.real)
    for i in range(pow_filt.shape[1]):
        pow_diff[:, i, ...] = 10 * np.log10(np.abs(np.divide(pow_filt[:, i, ...], pow_orig)).real)

    pow_diff_median = np.median(pow_diff, axis=2)

    weighted_pow_medians_all.append(pow_diff_median.mean(axis=1))
    del intermediate_layer_model
    tf.keras.backend.clear_session()

weighted_pow_medians_all = np.squeeze(np.asarray(weighted_pow_medians_all))

plt.figure(dpi=300)
sns.lineplot(x=(np.ones([weighted_pow_medians_all.shape[0], 1]) * np.expand_dims(f, 0)).flatten(),
             y=weighted_pow_medians_all.flatten(), errorbar='sd', color='purple')
plt.xticks(np.arange(0, 55, 5))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Temporal convolution frequency response')
# plt.savefig('F:/maryam_sh/General model/plots/Temporal convolution frequency response')

plt.show()
