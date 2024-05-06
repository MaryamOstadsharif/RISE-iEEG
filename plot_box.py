import matplotlib.pyplot as plt
import numpy as np

# save path
sp = "F:/maryam_sh/General model/plots/new/comparison_models_"
# 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
task = 'Singing_Music'
# number of folds for dataset: 'audio_visual' & 'under_sampling'=20 ,
# 'audio_visual' & 'under_sampling'=10,
# for dataset: 'music_reconstruction':29
num_fold = 29
# 'under_sampling' or 'over_sampling' or 'no_balancing'
balancing = 'over_sampling'
compare = True
# compare_type: 'patient' or 'fold'
compare_type = 'fold'
# reg_coef: 0.01, 0.05, 0.1
reg_coef = 0.05
# reg: 'L1' ,'L2'
reg = 'L2'

models = []
data = {}
if compare:
    models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']
    for model_type in models:
        data[model_type] = []
        if compare_type == 'patient':
            data_one = np.load(
                'F:/maryam_sh/Htnet_mydata_2/results/' + task + '/' + balancing + '/2024-05-02-18-45-25/accuracy/combined_sbjs_power'
                                                                                  '/fscore_gen_each_patient' + model_type + str(
                    num_fold) + '.npy')
            data[model_type] = np.mean(data_one, axis=0)
        else:
            data_one = np.load(
                'F:/maryam_sh/Htnet_mydata_2/results/' + task + '/' + balancing + '/2024-02-18-10-23-31/accuracy/combined_sbjs_power'
                                                                                  '/fscore_gen_' + model_type + str(
                    num_fold) + '.npy')
            data[model_type] = data_one[:, 2]

num_fold = 10
models = ['ECoGNet_'] + models
if compare_type == 'patient':
    data_one = np.load(
        'F:/maryam_sh/General model/General code/results/' + task + '/' + balancing + '/2024-05-04-11-45-22/accuracy'
                                                                                      '/fscore_' + models[
            0] + 'each_patient_' + str(
            num_fold) + '.npy')
    data[models[0]] = np.mean(data_one, axis=0)
else:
    data_one = np.load(
        'F:/maryam_sh/General model/General code/results/' + task + '/' + balancing + '/2024-05-05-13-25-48/accuracy'
                                                                                      '/fscore_' + models[0] + str(
            num_fold) + '.npy')
    data[models[0]] = data_one[:10, 2]

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
color_list = ['yellow', 'cyan', 'lime', 'blue', 'violet', 'darkgray', 'pink', 'darkgray', 'orangered', 'deeppink']

for i, model_type in enumerate(models):
    box1 = ax.boxplot(data[model_type], patch_artist=True, positions=[i], widths=0.6,
                      showfliers=False)  # Adjust the width and position as needed
    for box_element in box1['boxes']:
        box_element.set(color='black', alpha=0.7)
        box_element.set_facecolor(color_list[i])  # Set the facecolor

    for k in range(len(data[model_type])):
        ax.plot(i, data[model_type][k], 'o', markersize=4, color='red')

# Customize the plot
ax.set_xticks([i for i in range(len(models))])
if task == 'Singing_Music' or task == 'move_rest':
    ax.set_xticklabels(['ECoGNet', 'HTNet', 'EEGNet', 'RandomForest', 'MinimumDistance'], fontsize=14)
else:
    ax.set_xticklabels(['GN_CNN'])
ax.set_ylabel('F-score', fontsize=15)
ax.set_title(f'Model version 2 (add {reg} regularization {reg_coef}) / each point, each {compare_type}', fontsize=14)
plt.ylim(0.7, 0.9)

# Show the plot
# plt.show()
fig.savefig(sp + task + '_' + balancing + '_' + compare_type + f'_fscore_{reg}_{int(reg_coef * 100)}')
fig.savefig(sp + task + '_' + balancing + '_' + compare_type + f'_fscore_{reg}_{int(reg_coef * 100)}.svg')
