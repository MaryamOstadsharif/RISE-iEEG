### Transfer learning
import numpy as np
import matplotlib.pyplot as plt

num_patient = 9

bar_pos = np.arange(num_patient) * 4

ECoGNet_acc = np.mean(np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
                              "2024-03-10-12-27-29/accuracy/fscore_GNCNN_each_patient_5.npy"), axis=0)

data = np.load("F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/"
               "2024-03-10-12-27-29/accuracy/fscore_GNCNN_each_patient_5.npy")
for i in range(data.shape[1]):
    print(f'Patient {i} : {round(np.mean(data, axis=0)[i] * 100, 2)}+{round(np.std(data, axis=0)[i] * 100, 2)}')

d = [4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2, 16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23]
g = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
pos = []
for j in g:
    for i, num in enumerate(d):
        if d[i] == j:
            pos.append(i)

HTNet_acc = \
np.load("F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/2024-03-08-21-21-09/"
        "accuracy/acc_gen_eegnet_hilb_29.npy")[:, 2][pos]
EEGNet_acc = \
np.load("F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/2024-03-08-21-21-09/"
        "accuracy/acc_gen_eegnet_29.npy")[:, 2][pos]
rf_acc = np.load("F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/2024-03-08-21-21-09/"
                 "accuracy/acc_gen_rf_29.npy")[:, 2][pos]
rieman_acc = \
np.load("F:/maryam_sh/new_dataset/code/results/Singing_Music/Multi_patient/over_sampling/2024-03-08-21-21-09/"
        "accuracy/acc_gen_riemann_29.npy")[:, 2][pos]

print(F' ECoGNet_acc : {round(np.mean(ECoGNet_acc * 100), 2)} ± {round(np.std(ECoGNet_acc * 100), 2)}')
print(F' EEGNet_acc : {round(np.mean(EEGNet_acc * 100), 2)} ± {round(np.std(EEGNet_acc * 100), 2)}')
print(F' HTNet_acc : {round(np.mean(HTNet_acc * 100), 2)} ± {round(np.std(HTNet_acc * 100), 2)}')
print(F' rf_acc : {round(np.mean(rf_acc * 100), 2)} ± {round(np.std(rf_acc * 100), 2)}')
print(F' rieman_acc : {round(np.mean(rieman_acc * 100), 2)} ± {round(np.std(rieman_acc * 100), 2)}')

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
plt.bar(bar_pos, ECoGNet_acc, color='red', width=0.5, edgecolor='gray', label='ECoGNet', alpha=0.7)
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
plt.ylim(0.1, 1.1)
plt.ylabel('F-score', fontsize=15)
plt.xlabel('Patient ID', fontsize=15)
# plt.title(f'Classification unseen patients', fontsize=9)
plt.legend(ncol=2, fontsize=10)
plt.savefig("F:/maryam_sh/General model/plots/comparison_models.png")
plt.savefig("F:/maryam_sh/General model/plots/comparison_models.svg")
plt.legend()

######################################################################################################
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
### ECoGNet_multi vs Single_model_combined channel
