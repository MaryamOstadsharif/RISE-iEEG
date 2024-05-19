# ## Transfer learning
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Task: 'Singing_Music' & 'move_rest'
task = 'move_rest'
# 'matplot' or 'sns'
use_lib = 'sns'

if task == 'move_rest':
    num_patient = 12
    bar_pos = np.arange(num_patient) * 4
    ECoGNet_acc = np.zeros(num_patient)
    path_ecog = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/transfer/'
    for i in range(num_patient):
        data = np.mean(np.load(path_ecog + str(i) + "/acc_ECoGNet_5.npy")[-3:, 2])
        ECoGNet_acc[i] = data

    paths = ['F:/maryam_sh/HTNet model/move_rest_ecog/3/','F:/maryam_sh/HTNet model/move_rest_ecog/5/',
            'F:/maryam_sh/HTNet model/move_rest_ecog/combined_sbjs_power/']

    HTNet_acc_all = np.zeros((12, 3))
    EEGNet_acc_all = np.zeros((12, 3))
    rf_acc_all = np.zeros((12, 3))
    rieman_acc_all = np.zeros((12, 3))
    for k,path in enumerate(paths):
        data_htnet= np.load(path + "acc_gen_eegnet_hilb_12.npy")[:, 2]
        data_EEGNet= np.load(path + "acc_gen_eegnet_12.npy")[:, 2]
        data_rf= np.load(path + "acc_gen_rf_12.npy")[:, 2]
        data_rieman= np.load(path + "acc_gen_riemann_12.npy")[:, 2]

        pos = np.array([5, 3, 1, 0, 4, 6, 10, 2, 9, 11, 8, 7])
        for i in range(12):
            HTNet_acc_all[i,k] = data_htnet[np.where(pos == i)]
            EEGNet_acc_all[i,k] = data_EEGNet[np.where(pos == i)]
            rf_acc_all[i,k] = data_rf[np.where(pos == i)]
            rieman_acc_all[i,k] = data_rieman[np.where(pos == i)]

    EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
    HTNet_acc = np.mean(HTNet_acc_all, axis=1)
    rf_acc = np.mean(rf_acc_all, axis=1)
    rieman_acc = np.mean(rieman_acc_all, axis=1)

    print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
    print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
    print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
    print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
    print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')

    #

    if use_lib == 'matplot':
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        plt.bar(bar_pos, ECoGNet_acc, color='red', width=0.5, edgecolor='gray', label='RISE-iEEG', alpha=0.7)
        bar_pos = bar_pos + 0.5
        plt.bar(bar_pos, HTNet_acc, color='yellow', width=0.5, edgecolor='gray', label='HTNet', alpha=0.7)
        bar_pos = bar_pos + 0.5
        plt.bar(bar_pos, EEGNet_acc, color='cyan', width=0.5, edgecolor='gray', label='EEGNet', alpha=0.7)
        bar_pos = bar_pos + 0.5
        plt.bar(bar_pos, rf_acc, color='deeppink', width=0.5, edgecolor='gray', label='Random Forest', alpha=0.7)
        bar_pos = bar_pos + 0.5
        plt.bar(bar_pos, rieman_acc, color='blue', width=0.5, edgecolor='gray', label='Minimum Distance', alpha=0.7)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(bar_pos - 1, [f'P{i + 1}' for i in range(num_patient)])
        plt.yticks(np.arange(0.1, 1.1, 0.1))
        plt.ylim(0, 1.1)
        plt.ylabel('F1 score', fontsize=15)
        plt.xlabel('Patient ID', fontsize=15)
        # plt.title(f'Classification unseen patients', fontsize=9)
        plt.legend(ncol=2, fontsize=10)
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move rest.png")
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move rest.svg")
        plt.legend()
    if use_lib == 'sns':
        # Create a DataFrame for easier plotting with seaborn
        data = {
            'Patient': np.tile(np.arange(1, num_patient + 1), 5),
            'Accuracy': np.concatenate([ECoGNet_acc, HTNet_acc, EEGNet_acc, rf_acc, rieman_acc]),
            'Model': np.repeat(['RISE-iEEG', 'HTNet', 'EEGNet', 'Random Forest', 'Minimum Distance'], num_patient)
        }
        df = pd.DataFrame(data)
        # Define a custom color palette
        custom_palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']

        # Plotting with seaborn
        plt.figure(figsize=(12, 6), dpi=300)
        ax = sns.barplot(x='Patient', y='Accuracy', hue='Model', data=df, palette=custom_palette)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=8)

        plt.yticks(np.arange(0.1, 1.1, 0.1))
        plt.ylim(0, 1.1)
        plt.ylabel('F1 score', fontsize=15)
        plt.xlabel('Patient ID', fontsize=15)
        plt.legend(ncol=2, fontsize=10)
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest_sns_v2.png")
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest_sns_v2.svg")

if task == 'Singing_Music':
    num_patient = 29

    bar_pos = np.arange(num_patient) * 3.5

    ECoGNet_acc = np.zeros(num_patient)
    path_ecog = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/transfer/n fold/'
    for i in range(num_patient):
        data = np.mean(np.load(path_ecog + str(i) + "/fscore_ECoGNet_3.npy")[:, 2])
        ECoGNet_acc[i] = data

    base_paths = 'F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/'
    paths = [base_paths+'2024-03-18-03-06-57/accuracy/',
             base_paths+'2024-05-18-17-53-30/accuracy/',
             base_paths+'2024-05-18-23-18-14/accuracy/']
    pos = np.array(
        [4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2, 16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23])

    HTNet_acc_all = np.zeros((num_patient, 3))
    EEGNet_acc_all = np.zeros((num_patient, 3))
    rf_acc_all = np.zeros((num_patient, 3))
    rieman_acc_all = np.zeros((num_patient, 3))
    for k, path in enumerate(paths):
        data_htnet = np.load(path + "fscore_gen_eegnet_hilb_29.npy")[:, 2]
        data_EEGNet = np.load(path + "fscore_gen_eegnet_29.npy")[:, 2]
        data_rf = np.load(path + "fscore_gen_rf_29.npy")[:, 2]
        data_rieman = np.load(path + "fscore_gen_riemann_29.npy")[:, 2]

        for i in range(num_patient):
            HTNet_acc_all[i, k] = data_htnet[np.where(pos == i)]
            EEGNet_acc_all[i, k] = data_EEGNet[np.where(pos == i)]
            rf_acc_all[i, k] = data_rf[np.where(pos == i)]
            rieman_acc_all[i, k] = data_rieman[np.where(pos == i)]

    EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
    HTNet_acc = np.mean(HTNet_acc_all, axis=1)
    rf_acc = np.mean(rf_acc_all, axis=1)
    rieman_acc = np.mean(rieman_acc_all, axis=1)

    print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
    print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
    print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
    print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
    print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')

    if use_lib == 'matplot':
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        barwidth = 0.5
        plt.bar(bar_pos, ECoGNet_acc, color='#E91E63', width=0.5, edgecolor='gray', label='RISE-iEEG', alpha=0.7)
        bar_pos = bar_pos + barwidth
        plt.bar(bar_pos, HTNet_acc, color='#3357FF', width=0.5, edgecolor='gray', label='HTNet', alpha=0.7)
        bar_pos = bar_pos + barwidth
        plt.bar(bar_pos, EEGNet_acc, color='cyan', width=0.5, edgecolor='gray', label='EEGNet', alpha=0.7)
        bar_pos = bar_pos + barwidth
        plt.bar(bar_pos, rf_acc, color='#8E44AD', width=0.5, edgecolor='gray', label='Random Forest', alpha=0.7)
        bar_pos = bar_pos + barwidth
        plt.bar(bar_pos, rieman_acc, color='limegreen', width=0.5, edgecolor='gray', label='Minimum Distance', alpha=0.7)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(bar_pos - 1, [f'P{i + 1}' for i in range(num_patient)], fontsize=8)
        plt.yticks(np.arange(0.1, 1.1, 0.1))
        plt.ylim(0, 1.2)
        plt.ylabel('F-score', fontsize=15)
        plt.xlabel('Patient ID', fontsize=15)
        # plt.title(f'Classification unseen patients', fontsize=9)
        plt.legend(ncol=2, fontsize=10)
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing music.png")
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing music.svg")
        plt.legend()

    if use_lib == 'sns':
        # Create a DataFrame for easier plotting with seaborn
        data = {
            'Patient': np.tile(np.arange(1, num_patient + 1), 5),
            'Accuracy': np.concatenate([ECoGNet_acc, HTNet_acc, EEGNet_acc, rf_acc, rieman_acc]),
            'Model': np.repeat(['RISE-iEEG', 'HTNet', 'EEGNet', 'Random Forest', 'Minimum Distance'], num_patient)
        }
        df = pd.DataFrame(data)
        # Define a custom color palette
        custom_palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']

        # Plotting with seaborn
        plt.figure(figsize=(12, 6), dpi=300)
        ax = sns.barplot(x='Patient', y='Accuracy', hue='Model', data=df, palette=custom_palette)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=8)
        plt.yticks(np.arange(0.1, 1.1, 0.1))
        plt.ylim(0, 1.2)
        plt.ylabel('F1 score', fontsize=15)
        plt.xlabel('Patient ID', fontsize=15)
        plt.legend(ncol=2, fontsize=10)
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music_sns_v2.png")
        plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music_sns_v2.svg")

#######################################################################################################
# plot each point for decoders
# import numpy as np
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(12,5),dpi=300)
#
# path = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-05-10-11-19-33/accuracy/'
# data = np.load(path + "fscore_ECoGNet_each_patient_10.npy")
#
# # print(f'ECoGNet: \n{np.mean(data,axis=0)}\n')
# plt.plot(np.mean(data,axis=0), 'o', label='RISE-iEEG', color='blue')
#
# path = 'F:/maryam_sh/Htnet_mydata_2/results/move_rest/no_balancing/2024-05-09-20-17-05/accuracy/combined_sbjs_power/'
#
# data = np.load(path + "fscore_gen_each_patienteegnet_hilb_10.npy")
# # print(f'eegnet_hilb: \n{np.mean(data,axis=0)}\n')
# plt.plot(np.mean(data,axis=0), 'o', label='HTNet', color='gold')
#
# data = np.load(path + "fscore_gen_each_patienteegnet_10.npy")
# # print(f'eegnet: \n{np.mean(data,axis=0)}\n')
# plt.plot(np.mean(data,axis=0), 'o', label='EEGNet', color='lime')
#
# data = np.load(path + "fscore_gen_each_patientrf_10.npy")
# # print(f'Rf: \n{np.mean(data,axis=0)}\n')
# plt.plot(np.mean(data,axis=0), 'o', label='RandomForest', color='purple')
#
# data = np.load(path + "fscore_gen_each_patientriemann_10.npy")
# # print(f'riemann: \n{np.mean(data,axis=0)}\n')
# plt.plot(np.mean(data,axis=0), 'o', label='Minimum Distance', color='cyan')
#
# plt.ylim(0.3,1)
# plt.xlim(-1,12)
# plt.xticks(np.arange(12), [f'P{i + 1}' for i in range(12)], fontsize=8)
# plt.legend(ncol=2)
# plt.xlabel('Patient ID', fontsize=15)
# ax.set_ylabel('Test F1 score', fontsize=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_move rest.png")
# plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_move rest.svg")
#

###############################################################################################################
### ECoGNet_multi vs ECoGNet_single
# import numpy as np
# import matplotlib.pyplot as plt
#
# num_patient=29
#
# bar_pos=np.arange(num_patient)*2
#
# # ECoGNet_multi=np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
# #                       "2024-01-27-13-30-18/accuracy/fscore_gen_GNCNN_each_patient_29.npy")
# ECoGNet_multi=np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
#                       "2024-03-06-12-16-57/accuracy/fscore_GNCNN_each_patient_5.npy")
#
# ECoGNet_single =[]
# for i in range(num_patient):
#     ECoGNet_single.append(np.mean(np.load(f"F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/train_test_each_patient/{i}/"
#                                 "/accuracy/fscore_GNCNN_each_patient_5.npy")))
#
# print(np.mean(ECoGNet_single))
# print(np.mean(ECoGNet_multi))
#
# fig, ax = plt.subplots(figsize=(15,5),dpi=300)
# plt.bar(bar_pos,  np.mean(ECoGNet_multi,axis=0), color='deeppink', width=0.5,edgecolor='gray', label='ECoGNet(multi)', alpha=0.7)
# bar_pos=bar_pos+0.5
# plt.bar(bar_pos, ECoGNet_single, color='cyan', width=0.5,edgecolor='gray', label='ECoGNet(single)', alpha=0.7)
#
# plt.xticks(bar_pos - 0.25, [f'P{i+1}' for i in range(num_patient)])
# plt.yticks(np.arange(0.1, 1.1, 0.1))
# plt.ylim(0.5,1)
# plt.ylabel('F-score', fontsize=15)
# plt.xlabel('Patient ID', fontsize=15)
# plt.title(f'Compare ECoGNet_multi vs ECoGNet_single', fontsize=15)
# plt.legend(fontsize=12)
# plt.savefig("F:/maryam_sh/General model/plots/compare ECoGNet_multi vs ECoGNet_single8")

######################################################################################################
### ECoGNet_multi vs Single_model_combined channel

# import numpy as np
# import matplotlib.pyplot as plt
#
# num_patient = 29
# # 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
# task = 'Singing_Music'
# #*7
# bar_pos = np.arange(num_patient) * 2
# #7
# width_bar=0.5
# ECoGNet_multi = np.load("F:/maryam_sh/General model/General code/results/"+task+"/over_sampling/"
#                         "2024-03-06-12-16-57/accuracy/fscore_GNCNN_each_patient_5.npy")
# #Singing_Music: 2024-03-06-12-16-57
# #Speech_Music:2024-03-16-15-15-23
# #Question_Answer:2023-12-22-03-04-25_star
#
# combined_channel = np.load("F:/maryam_sh/ziaei_github/iEEG_fMRI_audiovisual/results/"+task+"/oversampling/"
#                            "2024-03-06-21-53-30/plots/classification/SVM_over_sampling/Max_voting/f_measure_all_ensemble.npy")
#
# #Singing_Music: 2024-03-06-21-53-30
# #Speech_Music:2023-12-18-22-51-30
# #Question_Answer:2023-12-16-13-34-16
#
# single_channel = np.load("F:/maryam_sh/ziaei_github/iEEG_fMRI_audiovisual/results/"+task+"/oversampling/"
#                          "2024-03-06-21-53-30/plots/classification/SVM_over_sampling/Max_voting/max_performance_all.npy")
#
# fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
# plt.bar(bar_pos, np.mean(ECoGNet_multi, axis=0), color='cyan', width=width_bar, edgecolor='gray', label='ECoGNet', alpha=0.7)
# bar_pos = bar_pos + 0.5
# plt.bar(bar_pos, combined_channel, color='deeppink', width=width_bar, edgecolor='gray', label='Single_Patient(Combined_channel)',
#         alpha=0.7)
# bar_pos = bar_pos + 0.5
# plt.bar(bar_pos, single_channel, color='yellow', width=width_bar, edgecolor='gray', label='Single_Patient(Single_channel)',
#         alpha=0.7)
#
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# plt.xticks(bar_pos - 0.25, [f'P{i + 1}' for i in range(num_patient)])
# plt.yticks(np.arange(0.1, 1.1, 0.1))
# plt.ylim(0.4, 1.1)
# plt.ylabel('F-score', fontsize=15)
# plt.xlabel('Patient ID', fontsize=15)
# # plt.title(f'Compare ECoGNet vs Combined_channel', fontsize=15)
# plt.legend(loc='upper right')
#
# plt.savefig(f"F:/maryam_sh/General model/plots/compare ECoGNet_Combined_single_{task}")
# plt.savefig(f"F:/maryam_sh/General model/plots/compare ECoGNet_Combined_single_{task}.svg")

############################################################################################################
