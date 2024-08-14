import matplotlib.pyplot as plt
import numpy as np

# Save path for the plots
save_path = "F:/maryam_sh/General model/plots/comparison_models_"
path_save_comparison_model = {'Singing_Music': "F:/maryam_sh/General model/General code/results/Singing_Music/"
                                               "over_sampling/2024-05-12-02-51-36/accuracy/combined_sbjs_power",
                              'move_rest': "F:/maryam_sh/General model/General code/results/move_rest/"
                                           "no_balancing/2024-05-09-20-17-05/accuracy/combined_sbjs_power"}
path_save_RISEiEEG_model = {'Singing_Music': "F:/maryam_sh/General model/General code/results/Singing_Music/"
                                             "over_sampling/2024-05-11-00-31-29/accuracy",
                            'move_rest': "F:/maryam_sh/General model/General code/results/move_rest/"
                                         "no_balancing/2024-06-04-20-49-20/accuracy"}
# Number of folds
num_fold = 10
# List of model types to compare
models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']
# Dictionary to store the data for each model
f1_score = {'Singing_Music': {}, 'move_rest': {}}

for model_type in models:
    for task in f1_score.keys():
        f1_score[task][model_type] = np.load(
            path_save_comparison_model[task] + '/fscore_gen_' + model_type + str(num_fold) + '.npy')[:, 2]

models = ['RISEiEEG_'] + models
for task in f1_score.keys():
    f1_score[task][models[0]] = np.load(path_save_RISEiEEG_model + '/fscore_' + models[0] + str(num_fold) + '.npy')[:,
                                2]

# Create a figure and axis for the plot
fig, ax = plt.subplots(ncols=2, figsize=(12, 6), dpi=300)
# List of colors for the box plots
color_list = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
pos_box = [0, 0.3, 0.6, 0.9, 1.2]

# Plot box for each model
for task in f1_score.keys():
    for i, model_type in enumerate(models):
        box1 = ax[task].boxplot(f1_score[task][model_type], patch_artist=True, positions=[pos_box[i]], widths=0.25,
                                showfliers=False)
        for box_element in box1['boxes']:
            box_element.set_edgecolor('black')
            box_element.set(color='black', alpha=0.7)
            box_element.set_facecolor(color_list[i])
        for median in box1['medians']:
            median.set_color('black')
        for k in range(num_fold):
            ax[task].plot(pos_box[i], f1_score[task][model_type][k], 'o', markersize=3, color='black')

    ax[task].set_xticks(pos_box)
    ax[task].set_xticklabels(['RISE-iEEG', 'HTNet', 'EEGNet', 'RF', 'MD'], fontsize=18, rotation=-45)
    ax[task].set_xlim(-0.2, 1.4)
    ax[task].spines['right'].set_visible(False)
    ax[task].spines['top'].set_visible(False)
    ax[task].set_ylabel('F1 score', fontsize=18)
    ax[task].tick_params(axis='y', which='major', labelsize=14)
    if task == 0:
        ax[task].set_yticks(np.arange(0.5, 0.95, 0.05), fontsize=16)
        ax[task].set_ylim(0.5, 0.9)
    else:
        ax[task].set_yticks(np.arange(0.5, 0.8, 0.05), fontsize=16)
        ax[task].set_ylim(0.49, 0.75)

plt.tight_layout()
fig.savefig(save_path + f'Comparison_same_patient.png')
fig.savefig(save_path + f'Comparison_same_patient.svg')
