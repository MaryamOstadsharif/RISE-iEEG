# import numpy as np
# import matplotlib.pyplot as plt
# from prettytable import PrettyTable
#
# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-05-10-11-19-33/accuracy/'
# data_1 = np.mean(np.load(path + "fscore_ECoGNet_each_patient_10.npy"), axis=0)
#
# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-05-11-11-43-19/accuracy/'
# data_5 = np.mean(np.load(path + "fscore_ECoGNet_each_patient_10.npy"), axis=0)
#
# row_labels=[]
# for i in range(29):
#     # print("Test accuracy of fold", str(i), "= ", np.mean(acc_GN,axis=0)[i],'/', acc_single[i])
#     row_labels.append('fold_'+str(i))
#
# table = PrettyTable()
# table.field_names = ["Fold", "Mean accuracy (L2, 0.01)", "Mean accuracy (L2, 0.05)"]
#
# for i in range(12):
#     table.add_row([f"patient_{i+1}", round(data_1[i]*100,2), round(data_5[i]*100,2)])
#
# # Print PrettyTable
# print(table)
# #
# print('end')

#################################################################################################################

# import numpy as np
#
# acc_GN = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-10-12-27-29/accuracy/fscore_GNCNN_each_patient_5.npy")
#
# for i in range(9):
#     print("Test accuracy of patient", str(i), "= ", np.mean(acc_GN,axis=0)[i]*100)
#
#
# print('end')


#################################################################################################################
# import joblib
# #
# # path ='F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-03-02-12-48-03/accuracy/optuna/optuna_study.pkl'
# # # Load the study object from the pickle file
# # study1 = joblib.load(path)
# #
# # path ='F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-03-03-09-28-09/accuracy/optuna/optuna_study.pkl'
# # # Load the study object from the pickle file
# # study2  = joblib.load(path)
#
# path ='F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-03-13-10-14-16/accuracy/optuna/optuna_study.pkl'
# # Load the study object from the pickle file
# study3 = joblib.load(path)
# #
# # # Now you can access various attributes and results of the study object
# # # For example, you can print the best parameters found during optimization:
# # print("Best parameters:", study1.best_params)
# # print("Best parameters:", study2.best_params)
# print("Best parameters:", study3.best_params)
#
# print('end')

#######################################################################################################################
#
# import numpy as np
#
# acc_GN = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-12-18-35/accuracy/fscore_GNCNN_each_patient_5.npy")
# acc_single=[]
# acc_single.append(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-10-41-27/accuracy/fscore_GNCNN_each_patient_5.npy"))
#
# acc_single.append(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-13-40-07/accuracy/fscore_GNCNN_each_patient_5.npy"))
#
# acc_single.append(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-13-44-18/accuracy/fscore_GNCNN_each_patient_5.npy"))
#
# acc_single.append(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-13-49-41/accuracy/fscore_GNCNN_each_patient_5.npy"))
#
# acc_single.append(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                    "2024-03-11-14-12-37/accuracy/fscore_GNCNN_each_patient_5.npy"))
#
# for i in range(5):
#     print(F"Test accuracy of patient_{i}= {np.mean(acc_GN,axis=0)[i]}/ {np.mean(acc_single[i])}")
#
#
# print('end')


# ##############################################
# import tensorflow as tf
#
# pretrained_model = tf.keras.models.load_model('F:/maryam_sh/General model/General code/results/Singing_Music/'
#                                               'over_sampling/transfer/2/accuracy/checkpoint_gen__fold0.h5')
# num_input_pretrained_model=len(pretrained_model.input_names)
#
# print('end')

###################################################
# import numpy as np
# from scipy.fftpack import fft
# from scipy.signal import hann
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D
# from keras import backend as K
# import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU
#
# weighted_pow_medians_all = []
# for curr_fold in range(5):
#     loadname = 'F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/2024-03-18-03-06-57/' \
#                'accuracy/checkpoint_gen_eegnet_fold'+str(curr_fold)+'.h5'
#
#     model_curr = tf.keras.models.load_model(loadname)
#
#     output_layer = 'conv2d'
#     srate_new = 100
#     intermediate_layer_model = Model(inputs=[model_curr.input[0]], outputs=[model_curr.get_layer(output_layer).output])
#
#     nrows = int(intermediate_layer_model.input[0].shape[-2])
#     ncols = int(intermediate_layer_model.input[0].shape[-1])
#     data_in = np.zeros([1, 1, nrows, ncols])
#     for i in range(nrows):
#         data_in[..., i, :] = np.random.standard_normal(ncols)
#     filt_dat = intermediate_layer_model.predict(data_in)  # error pops up if running on CPU
#
#     w = hann(data_in.shape[-1])
#     pow_orig = fft(data_in * w)
#     pow_filt = fft(filt_dat * w)
#     f = np.fft.fftfreq(pow_orig.shape[-1], d=1 / srate_new)
#
#     pow_orig = pow_orig[..., f > 0]
#     pow_filt = pow_filt[..., f > 0]
#     f = f[f > 0]
#     pow_diff = np.zeros_like(pow_filt.real)
#
#     for i in range(pow_filt.shape[1]):
#         pow_diff[:, i, ...] = 10 * np.log10(np.abs(np.divide(pow_filt[:, i, ...], pow_orig)).real)
#
#     pow_diff_median = np.median(pow_diff, axis=2)
#
#     weighted_pow_medians_all.append(pow_diff_median.mean(axis=1))
#     del intermediate_layer_model
#     tf.keras.backend.clear_session()
#
# weighted_pow_medians_all = np.squeeze(np.asarray(weighted_pow_medians_all))
#
# plt.figure(dpi=300)
# sns.lineplot(x=(np.ones([weighted_pow_medians_all.shape[0], 1]) * np.expand_dims(f, 0)).flatten(),
#              y=weighted_pow_medians_all.flatten(), errorbar='sd', color='purple')
# plt.xticks(np.arange(0, 55, 5))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power (dB)')
# plt.title('Temporal convolution frequency response')
#
# plt.show()

######################################################################
# import numpy as np
#
# data_old = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-05-06-10-47-39/"
#                "accuracy/fscore_GNCNN_3.npy")
#
# data_new = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-05-06-10-40-31/"
#                "accuracy/fscore_GNCNN_3.npy")
#
# print(f'data_old: {np.mean(data_old[:,2])}')
# print(f'\n \n data_new: {np.mean(data_new[:,2])}')


import numpy as np

# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-05-13-13-49-18/accuracy/'
# data = np.load(path + "fscore_ECoGNet_10.npy")
# print(data[:,2])

# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-05-13-19-46-24/accuracy/'
# data = np.load(path + "fscore_ECoGNet_each_patient_10.npy")
# print(np.mean(data, axis=0))

# path = 'F:/maryam_sh/new_dataset/code/results/move_rest/Multi_patient/under_sampling/2024-05-13-20-41-04/accuracy/'
# data = np.load(path + "fscore_gen_eegnet_hilb_12.npy")
# print(data[:,2])
#
# path = 'F:/maryam_sh/new_dataset/code/results/move_rest/Multi_patient/under_sampling/2024-05-14-19-14-34/accuracy/'
# data = np.load(path + "acc_gen_eegnet_hilb_12.npy")
# print(data[:,2])
# print(np.mean(data[:,2]))

path = 'F:/maryam_sh/HTNet model/move_rest_ecog/combined_sbjs_power/'
data1 = np.load(path + "acc_gen_eegnet_12.npy")
print('\n',data1[:,2])
print(np.mean(data1[:,2]))

path = 'F:/maryam_sh/HTNet model/move_rest_ecog/5/'
data1 = np.load(path + "acc_gen_eegnet_12.npy")
print('\n',data1[:,2])
print(np.mean(data1[:,2]))

path = 'F:/maryam_sh/HTNet model/move_rest_ecog/4/'
data1 = np.load(path + "acc_gen_eegnet_12.npy")
print('\n',data1[:,2])
print(np.mean(data1[:,2]))

path = 'F:/maryam_sh/HTNet model/move_rest_ecog/3/'
data1 = np.load(path + "acc_gen_eegnet_12.npy")
print('\n',data1[:,2])
print(np.mean(data1[:,2]))

path = 'F:/maryam_sh/HTNet model/move_rest_ecog/1/'
data1 = np.load(path + "acc_gen_eegnet_hilb_12.npy")
print('\n',data1[:,2])
print(np.mean(data1[:,2]))

# print('\n\n\n')
# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/transfer2/'
# data_all=[]
# for i in range(12):
#     data = np.mean(np.load(path + str(i) + "/acc_ECoGNet_5.npy")[:,2])
#     print(data)
# print('\n\n\n')
#
# path = 'F:/maryam_sh/General model/General code/results/Question_Answer/over_sampling/2024-05-18-02-10-59/accuracy/'
# data1 = np.load(path + "fscore_ECoGNet_10.npy")
# print(data1[:,2])
# print(F' ECoGNet_acc : {round(np.mean(data1[:,2] * 100), 2)} ± {round(np.std(data1[:,2] * 100), 2)}')

print('end')

