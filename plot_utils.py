# # # ## Transfer learning
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Task: 'Singing_Music' & 'move_rest'
# task = 'move_rest'
# # 'matplot' or 'sns'
# use_lib = 'sns'
#
# if task == 'move_rest':
#     num_patient = 12
#     bar_pos = np.arange(num_patient) * 4
#     ECoGNet_acc = np.zeros(num_patient)
#     path_ecog = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/transfer2/'
#     for i in range(num_patient):
#         data = np.mean(np.load(path_ecog + str(i) + "/accuracy/acc_ECoGNet_3.npy")[:, 2])
#         ECoGNet_acc[i] = data
#
#     paths = ['F:/maryam_sh/HTNet model/move_rest_ecog/3/','F:/maryam_sh/HTNet model/move_rest_ecog/5/',
#             'F:/maryam_sh/HTNet model/move_rest_ecog/combined_sbjs_power/']
#
#     HTNet_acc_all = np.zeros((12, 3))
#     EEGNet_acc_all = np.zeros((12, 3))
#     rf_acc_all = np.zeros((12, 3))
#     rieman_acc_all = np.zeros((12, 3))
#     for k,path in enumerate(paths):
#         data_htnet= np.load(path + "acc_gen_eegnet_hilb_12.npy")[:, 2]
#         data_EEGNet= np.load(path + "acc_gen_eegnet_12.npy")[:, 2]
#         data_rf= np.load(path + "acc_gen_rf_12.npy")[:, 2]
#         data_rieman= np.load(path + "acc_gen_riemann_12.npy")[:, 2]
#
#         pos = np.array([5, 3, 1, 0, 4, 6, 10, 2, 9, 11, 8, 7])
#         for i in range(12):
#             HTNet_acc_all[i,k] = data_htnet[np.where(pos == i)]
#             EEGNet_acc_all[i,k] = data_EEGNet[np.where(pos == i)]
#             rf_acc_all[i,k] = data_rf[np.where(pos == i)]
#             rieman_acc_all[i,k] = data_rieman[np.where(pos == i)]
#
#     EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
#     HTNet_acc = np.mean(HTNet_acc_all, axis=1)
#     rf_acc = np.mean(rf_acc_all, axis=1)
#     rieman_acc = np.mean(rieman_acc_all, axis=1)
#
#     print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
#     print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
#     print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
#     print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
#     print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')
#
#     #
#
#     if use_lib == 'matplot':
#         fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
#         plt.bar(bar_pos, ECoGNet_acc, color='red', width=0.5, edgecolor='gray', label='RISE-iEEG', alpha=0.7)
#         bar_pos = bar_pos + 0.5
#         plt.bar(bar_pos, HTNet_acc, color='yellow', width=0.5, edgecolor='gray', label='HTNet', alpha=0.7)
#         bar_pos = bar_pos + 0.5
#         plt.bar(bar_pos, EEGNet_acc, color='cyan', width=0.5, edgecolor='gray', label='EEGNet', alpha=0.7)
#         bar_pos = bar_pos + 0.5
#         plt.bar(bar_pos, rf_acc, color='deeppink', width=0.5, edgecolor='gray', label='Random Forest', alpha=0.7)
#         bar_pos = bar_pos + 0.5
#         plt.bar(bar_pos, rieman_acc, color='blue', width=0.5, edgecolor='gray', label='Minimum Distance', alpha=0.7)
#
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         plt.xticks(bar_pos - 1, [f'P{i + 1}' for i in range(num_patient)])
#         plt.yticks(np.arange(0.1, 1.1, 0.1))
#         plt.ylim(0, 1.1)
#         plt.ylabel('F1 score', fontsize=15)
#         plt.xlabel('Patient ID', fontsize=15)
#         # plt.title(f'Classification unseen patients', fontsize=9)
#         plt.legend(ncol=2, fontsize=10)
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move rest.png")
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move rest.svg")
#         plt.legend()
#     if use_lib == 'sns':
#         # Create a DataFrame for easier plotting with seaborn
#         data = {
#             'Patient': np.tile(np.arange(1, num_patient + 1), 5),
#             'Accuracy': np.concatenate([ECoGNet_acc, HTNet_acc, EEGNet_acc, rf_acc, rieman_acc]),
#             'Model': np.repeat(['RISE-iEEG', 'HTNet', 'EEGNet', 'Random Forest', 'Minimum Distance'], num_patient)
#         }
#         df = pd.DataFrame(data)
#         # Define a custom color palette
#         custom_palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
#
#         # Plotting with seaborn
#         plt.figure(figsize=(12, 6), dpi=300)
#         ax = sns.barplot(x='Patient', y='Accuracy', hue='Model', data=df, palette=custom_palette)
#
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=12)
#
#         plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
#         plt.ylim(0, 1.1)
#         plt.ylabel('F1 score', fontsize=15)
#         plt.xlabel('Patient ID', fontsize=15)
#         plt.legend(ncol=2, fontsize=12)
#         plt.tight_layout()
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest.png")
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest.svg")
#
# if task == 'Singing_Music':
#     num_patient = 29
#
#     bar_pos = np.arange(num_patient) * 3.5
#
#     ECoGNet_acc = np.zeros(num_patient)
#     path_ecog = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/transfer/n fold/'
#     for i in range(num_patient):
#         data = np.mean(np.load(path_ecog + str(i) + "/fscore_ECoGNet_3.npy")[:, 2])
#         ECoGNet_acc[i] = data
#
#     base_paths = 'F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/'
#     paths = [base_paths+'2024-03-18-03-06-57/accuracy/',
#              base_paths+'2024-05-18-17-53-30/accuracy/',
#              base_paths+'2024-05-18-23-18-14/accuracy/']
#     pos = np.array(
#         [4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2, 16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23])
#
#     HTNet_acc_all = np.zeros((num_patient, 3))
#     EEGNet_acc_all = np.zeros((num_patient, 3))
#     rf_acc_all = np.zeros((num_patient, 3))
#     rieman_acc_all = np.zeros((num_patient, 3))
#     for k, path in enumerate(paths):
#         data_htnet = np.load(path + "fscore_gen_eegnet_hilb_29.npy")[:, 2]
#         data_EEGNet = np.load(path + "fscore_gen_eegnet_29.npy")[:, 2]
#         data_rf = np.load(path + "fscore_gen_rf_29.npy")[:, 2]
#         data_rieman = np.load(path + "fscore_gen_riemann_29.npy")[:, 2]
#
#         for i in range(num_patient):
#             HTNet_acc_all[i, k] = data_htnet[np.where(pos == i)]
#             EEGNet_acc_all[i, k] = data_EEGNet[np.where(pos == i)]
#             rf_acc_all[i, k] = data_rf[np.where(pos == i)]
#             rieman_acc_all[i, k] = data_rieman[np.where(pos == i)]
#
#     EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
#     HTNet_acc = np.mean(HTNet_acc_all, axis=1)
#     rf_acc = np.mean(rf_acc_all, axis=1)
#     rieman_acc = np.mean(rieman_acc_all, axis=1)
#
#     print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
#     print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
#     print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
#     print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
#     print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')
#
#     if use_lib == 'matplot':
#         fig, ax = plt.subplots(figsize=(6, 20), dpi=300)
#         barwidth = 0.5
#         plt.bar(bar_pos, ECoGNet_acc, color='#E91E63', width=0.5, edgecolor='gray', label='RISE-iEEG', alpha=0.7)
#         bar_pos = bar_pos + barwidth
#         plt.bar(bar_pos, HTNet_acc, color='#3357FF', width=0.5, edgecolor='gray', label='HTNet', alpha=0.7)
#         bar_pos = bar_pos + barwidth
#         plt.bar(bar_pos, EEGNet_acc, color='cyan', width=0.5, edgecolor='gray', label='EEGNet', alpha=0.7)
#         bar_pos = bar_pos + barwidth
#         plt.bar(bar_pos, rf_acc, color='#8E44AD', width=0.5, edgecolor='gray', label='Random Forest', alpha=0.7)
#         bar_pos = bar_pos + barwidth
#         plt.bar(bar_pos, rieman_acc, color='limegreen', width=0.5, edgecolor='gray', label='Minimum Distance', alpha=0.7)
#
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         plt.xticks(bar_pos - 1, [f'P{i + 1}' for i in range(num_patient)], fontsize=8)
#         plt.yticks(np.arange(0.1, 1.1, 0.1))
#         plt.ylim(0, 1.2)
#         plt.ylabel('F-score', fontsize=15)
#         plt.xlabel('Patient ID', fontsize=15)
#         # plt.title(f'Classification unseen patients', fontsize=9)
#         plt.legend(ncol=2, fontsize=10)
#         plt.tight_layout()
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing music_h.png")
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing music_h.svg")
#         plt.legend()
#
#     if use_lib == 'sns':
#         # Create a DataFrame for easier plotting with seaborn
#         data = {
#             'Patient': np.tile(np.arange(1, num_patient + 1), 5),
#             'Accuracy': np.concatenate([ECoGNet_acc, HTNet_acc, EEGNet_acc, rf_acc, rieman_acc]),
#             'Model': np.repeat(['RISE-iEEG', 'HTNet', 'EEGNet', 'Random Forest', 'Minimum Distance'], num_patient)
#         }
#         df = pd.DataFrame(data)
#         # Define a custom color palette
#         custom_palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
#
#         # # Plotting with seaborn
#         # if mode == 'hor':
#         #     fig, ax = plt.subplots(figsize=(10, 20), dpi=300)
#         #     ax = sns.barplot(x='Accuracy', y='Patient', hue='Model', data=df, palette=custom_palette, orient='h')
#         #
#         #     ax.spines['right'].set_visible(False)
#         #     ax.spines['top'].set_visible(False)
#         #     plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=20)
#         #     plt.yticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=20)
#         #     plt.xlim(0, 1.2)
#         #     plt.xlabel('F1 score', fontsize=25)
#         #     plt.ylabel('Patient ID', fontsize=25)
#         #     plt.legend(fontsize=15)
#         # else:
#         plt.figure(figsize=(12, 6), dpi=300)
#         ax = sns.barplot(x='Patient', y='Accuracy', hue='Model', data=df, palette=custom_palette)
#
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=10)
#         plt.yticks(np.arange(0.1, 1.1, 0.1), fontsize=12)
#         plt.ylim(0, 1.1)
#         plt.ylabel('F1 score', fontsize=15)
#         plt.xlabel('Patient ID', fontsize=15)
#         plt.legend(ncol=2, fontsize=12)
#         plt.tight_layout()
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music.png")
#         plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music.svg")
# ###############################################################################################
# Unseen patient, scatter plot
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Task: 'Singing_Music' & 'move_rest'
# task = 'Singing_Music'
#
# if task == 'Singing_Music':
#     num_patient = 29
#
#     ECoGNet_acc = np.zeros(num_patient)
#     path_ecog = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/transfer/n fold/'
#     for i in range(num_patient):
#         data = np.mean(np.load(path_ecog + str(i) + "/fscore_ECoGNet_3.npy")[:, 2])
#         ECoGNet_acc[i] = data
#
#     base_paths = 'F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/'
#     paths = [base_paths + '2024-03-18-03-06-57/accuracy/',
#              base_paths + '2024-05-18-17-53-30/accuracy/',
#              base_paths + '2024-05-18-23-18-14/accuracy/']
#     pos = np.array(
#         [4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2, 16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23])
#
#     HTNet_acc_all = np.zeros((num_patient, 3))
#     EEGNet_acc_all = np.zeros((num_patient, 3))
#     rf_acc_all = np.zeros((num_patient, 3))
#     rieman_acc_all = np.zeros((num_patient, 3))
#     for k, path in enumerate(paths):
#         data_htnet = np.load(path + "fscore_gen_eegnet_hilb_29.npy")[:, 2]
#         data_EEGNet = np.load(path + "fscore_gen_eegnet_29.npy")[:, 2]
#         data_rf = np.load(path + "fscore_gen_rf_29.npy")[:, 2]
#         data_rieman = np.load(path + "fscore_gen_riemann_29.npy")[:, 2]
#
#         for i in range(num_patient):
#             HTNet_acc_all[i, k] = data_htnet[np.where(pos == i)]
#             EEGNet_acc_all[i, k] = data_EEGNet[np.where(pos == i)]
#             rf_acc_all[i, k] = data_rf[np.where(pos == i)]
#             rieman_acc_all[i, k] = data_rieman[np.where(pos == i)]
#
#     EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
#     HTNet_acc = np.mean(HTNet_acc_all, axis=1)
#     rf_acc = np.mean(rf_acc_all, axis=1)
#     rieman_acc = np.mean(rieman_acc_all, axis=1)
#
#     print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
#     print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
#     print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
#     print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
#     print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')
#
#     fig, ax = plt.subplots(figsize=(12, 5),dpi=300)
#
#     plt.plot(ECoGNet_acc, 'D', label='RISE-iEEG', color='#E91E63', markersize=8)
#     plt.plot(HTNet_acc, 'o', label='HTNet', color='#3357FF', markersize=8)
#     plt.plot(EEGNet_acc, 's', label='EEGNet', color='cyan', markersize=8)
#     plt.plot(rf_acc, '^', label='Random Forest', color='#8E44AD', markersize=8)
#     plt.plot(rieman_acc, 'p', label='Minimum Distance', color='limegreen', markersize=8)
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=10)
#
#     plt.yticks(np.arange(0.4, 1.1, 0.1), fontsize=12)
#     plt.ylim(0.4, 1.1)
#     plt.ylabel('F1 score', fontsize=15)
#     plt.xlabel('Patient ID', fontsize=15)
#     plt.legend(ncol=2, fontsize=12)
#     plt.tight_layout()
#     plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music_scatter_point.png")
#     plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_sing_music_scatter_point.svg")
#
# else:
#     num_patient = 12
#
#     ECoGNet_acc = np.zeros(num_patient)
#     path_ecog = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/transfer2/'
#     for i in range(num_patient):
#         data = np.mean(np.load(path_ecog + str(i) + "/accuracy/acc_ECoGNet_3.npy")[:, 2])
#         ECoGNet_acc[i] = data
#
#     paths = ['F:/maryam_sh/HTNet model/move_rest_ecog/3/', 'F:/maryam_sh/HTNet model/move_rest_ecog/5/',
#              'F:/maryam_sh/HTNet model/move_rest_ecog/combined_sbjs_power/']
#
#     HTNet_acc_all = np.zeros((12, 3))
#     EEGNet_acc_all = np.zeros((12, 3))
#     rf_acc_all = np.zeros((12, 3))
#     rieman_acc_all = np.zeros((12, 3))
#     for k, path in enumerate(paths):
#         data_htnet = np.load(path + "acc_gen_eegnet_hilb_12.npy")[:, 2]
#         data_EEGNet = np.load(path + "acc_gen_eegnet_12.npy")[:, 2]
#         data_rf = np.load(path + "acc_gen_rf_12.npy")[:, 2]
#         data_rieman = np.load(path + "acc_gen_riemann_12.npy")[:, 2]
#
#         pos = np.array([5, 3, 1, 0, 4, 6, 10, 2, 9, 11, 8, 7])
#
#         for i in range(num_patient):
#             HTNet_acc_all[i, k] = data_htnet[np.where(pos == i)]
#             EEGNet_acc_all[i, k] = data_EEGNet[np.where(pos == i)]
#             rf_acc_all[i, k] = data_rf[np.where(pos == i)]
#             rieman_acc_all[i, k] = data_rieman[np.where(pos == i)]
#
#     EEGNet_acc = np.mean(EEGNet_acc_all, axis=1)
#     HTNet_acc = np.mean(HTNet_acc_all, axis=1)
#     rf_acc = np.mean(rf_acc_all, axis=1)
#     rieman_acc = np.mean(rieman_acc_all, axis=1)
#
#     print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
#     print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
#     print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
#     print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
#     print(F' rieman_acc : {round(np.mean(rie
#     man_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')
#
#     fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
#
#     plt.plot(ECoGNet_acc, 'D', label='RISE-iEEG', color='#E91E63', markersize=10)
#     plt.plot(HTNet_acc, 'o', label='HTNet', color='#3357FF', markersize=10)
#     plt.plot(EEGNet_acc, 's', label='EEGNet', color='cyan', markersize=10)
#     plt.plot(rf_acc, '^', label='Random Forest', color='#8E44AD', markersize=10)
#     plt.plot(rieman_acc, 'p', label='Minimum Distance', color='limegreen', markersize=10)
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=12)
#
#     plt.yticks(np.arange(0.4, 1.1, 0.1), fontsize=12)
#     plt.ylim(0.4, 1.1)
#     plt.ylabel('F1 score', fontsize=15)
#     plt.xlabel('Patient ID', fontsize=15)
#     plt.legend(ncol=2, fontsize=12)
#     plt.tight_layout()
#     plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest_scatter_point.png")
#     plt.savefig("F:/maryam_sh/General model/plots/new/comparison_models_move_rest_scatter_point.svg")
# #
#
#

# #######################################################################################################
# plot each point for decoders
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Task: 'Singing_Music' & 'move_rest'
# task = 'Singing_Music'
#
# if task == 'move_rest':
#     num_patient = 12
#     path_ecog = 'F:/maryam_sh/General model/General code/results/move_rest/no_balancing/2024-06-04-20-49-20/accuracy/'
#     ECoGNet_acc = np.mean(np.load(path_ecog + "fscore_ECoGNet_each_patient_10.npy"), axis=0)
#
#     path_oth_models = 'F:/maryam_sh/Htnet_mydata_2/results/move_rest/no_balancing/2024-05-09-20-17-05/accuracy/' \
#                       'combined_sbjs_power/'
#     HTNet_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patienteegnet_hilb_10.npy"), axis=0)
#     EEGNet_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patienteegnet_10.npy"), axis=0)
#     rf_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patientrf_10.npy"), axis=0)
#     rieman_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patientriemann_10.npy"), axis=0)
#
#
#     fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
#     #
#     plt.plot(ECoGNet_acc, 'o', label='RISE-iEEG', color='#E91E63', markersize=10)
#     plt.plot(HTNet_acc, 'o', label='HTNet', color='#3357FF', markersize=10)
#     plt.plot(EEGNet_acc, 'o', label='EEGNet', color='cyan', markersize=10)
#     plt.plot(rf_acc, 'o', label='Random Forest', color='#8E44AD', markersize=10)
#     plt.plot(rieman_acc, 'o', label='Minimum Distance', color='limegreen', markersize=10)
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=12)
#
#     plt.yticks(np.arange(0.2, 1.1, 0.1), fontsize=12)
#     plt.ylim(0.2, 1.1)
#     plt.ylabel('F1 score', fontsize=15)
#     plt.xlabel('Patient ID', fontsize=15)
#     plt.legend(ncol=2, fontsize=12)
#     plt.tight_layout()
#
#     plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_move rest.png")
#     plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_move rest.svg")
#
# else:
#     num_patient = 29
#
#     path_ecog = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-05-11-00-31-29/accuracy/'
#     ECoGNet_acc = np.mean(np.load(path_ecog + "fscore_ECoGNet_each_patient_10.npy"), axis=0)
#
#     path_oth_models = 'F:/maryam_sh/Htnet_mydata_2/results/Singing_Music/over_sampling/2024-05-12-02-51-36/accuracy/' \
#                       'combined_sbjs_power/'
#     HTNet_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patienteegnet_hilb_10.npy"), axis=0)
#     EEGNet_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patienteegnet_10.npy"), axis=0)
#     rf_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patientrf_10.npy"), axis=0)
#     rieman_acc = np.mean(np.load(path_oth_models + "fscore_gen_each_patientriemann_10.npy"), axis=0)
#
#     fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
#     #
#     plt.plot(ECoGNet_acc, 'o', label='RISE-iEEG', color='#E91E63', markersize=8)
#     plt.plot(HTNet_acc, 'o', label='HTNet', color='#3357FF', markersize=8)
#     plt.plot(EEGNet_acc, 'o', label='EEGNet', color='cyan', markersize=8)
#     plt.plot(rf_acc, 'o', label='Random Forest', color='#8E44AD', markersize=8)
#     plt.plot(rieman_acc, 'o', label='Minimum Distance', color='limegreen', markersize=8)
#
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.xticks(np.arange(num_patient), [f'P{i + 1}' for i in range(num_patient)], fontsize=10)
#
#     plt.yticks(np.arange(0.2, 1.1, 0.1), fontsize=10)
#     plt.ylim(0.2, 1.15)
#     plt.ylabel('F1 score', fontsize=15)
#     plt.xlabel('Patient ID', fontsize=15)
#     plt.legend(ncol=2, fontsize=12)
#     plt.tight_layout()
#
#     plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_singing music.png")
#     plt.savefig("F:/maryam_sh/General model/plots/new/acc_patients_singing music.svg")

#######################################################################################################
