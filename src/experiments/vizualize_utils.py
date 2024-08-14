import matplotlib.pyplot as plt
import numpy as np


def plot_box_compare_models(save_path, path_save_comparison_model, path_save_RISEiEEG_model):
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
        f1_score[task][models[0]] = np.load(path_save_RISEiEEG_model[task] + '/fscore_' + models[0] + str(num_fold) + '.npy')[:,2]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), dpi=300)
    # List of colors for the box plots
    color_list = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
    pos_box = [0, 0.3, 0.6, 0.9, 1.2]
    text_fig =['A)', 'B)']
    # Plot box for each model
    for t, task in enumerate(f1_score.keys()):
        for i, model_type in enumerate(models):
            box1 = ax[t].boxplot(f1_score[task][model_type], patch_artist=True, positions=[pos_box[i]], widths=0.25,
                                    showfliers=False)
            for box_element in box1['boxes']:
                box_element.set_edgecolor('black')
                box_element.set(color='black', alpha=0.7)
                box_element.set_facecolor(color_list[i])
            for median in box1['medians']:
                median.set_color('black')
            for k in range(num_fold):
                ax[t].plot(pos_box[i], f1_score[task][model_type][k], 'o', markersize=3, color='black')

        ax[t].set_xticks(pos_box)
        ax[t].set_xticklabels(['RISE-iEEG', 'HTNet', 'EEGNet', 'RF', 'MD'], fontsize=18, rotation=-45)
        ax[t].set_xlim(-0.2, 1.4)
        ax[t].spines['right'].set_visible(False)
        ax[t].spines['top'].set_visible(False)
        ax[t].set_ylabel('F1 score', fontsize=18)
        ax[t].tick_params(axis='y', which='major', labelsize=14)
        if t == 0:
            ax[t].set_yticks(np.arange(0.5, 0.95, 0.05), fontsize=16)
            ax[t].set_ylim(0.5, 0.9)
        else:
            ax[t].set_yticks(np.arange(0.5, 0.8, 0.05), fontsize=16)
            ax[t].set_ylim(0.49, 0.75)

        ax[t].annotate(text_fig[t], xy=(-0.02, 1.1), xycoords='axes fraction', fontsize=20, fontweight='bold',
                     ha='center', va='center')

    plt.tight_layout()
    fig.savefig(save_path + f'Comparison_same_patient.png')
    fig.savefig(save_path + f'Comparison_same_patient.svg')
