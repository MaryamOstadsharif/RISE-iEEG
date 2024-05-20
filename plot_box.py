import matplotlib.pyplot as plt
import numpy as np

# Save path for the plots
sp = "F:/maryam_sh/General model/plots/new/comparison_models_"
# Task: 'Singing_Music' & 'move_rest'
task = 'move_rest'
# Number of folds
num_fold = 10
# Balancing method: 'over_sampling' or 'no_balancing'
balancing = 'no_balancing'
# Comparison type: 'patient' or 'fold'
compare_type = 'fold'
# # Regularization coefficient: 0.01, 0.05, 0.1
reg_coef = 0.01
# Regularization type: 'L1' ,'L2'
reg = 'L2'

# List of model types to compare
models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']

# Dictionary to store the data for each model
data = {}
# 2024-05-12-02-51-36 for 'Singing_Music'
# 2024-05-09-20-17-05 for 'move_rest'

# Load data for each model type
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
            'F:/maryam_sh/Htnet_mydata_2/results/' + task + '/' + balancing + '/2024-05-09-20-17-05/accuracy/combined_sbjs_power'
                                                                              '/fscore_gen_' + model_type + str(
                num_fold) + '.npy')
        data[model_type] = data_one[:, 2]


# 2024-05-11-00-31-29 for 'Singing_Music'
# 2024-05-10-11-19-33 for 'move_rest'
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
        'F:/maryam_sh/General model/General code/results/' + task + '/' + balancing + '/2024-05-10-11-19-33/accuracy'
                                                                                      '/fscore_' + models[0] + str(
            num_fold) + '.npy')
    data[models[0]] = data_one[:, 2]

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
# List of colors for the box plots
color_list = ['yellow', 'cyan', 'lime', 'blue', 'violet', 'darkgray', 'pink', 'darkgray', 'orangered', 'deeppink']

# Plot box for each model
for i, model_type in enumerate(models):
    box1 = ax.boxplot(data[model_type], patch_artist=True, positions=[i], widths=0.6, showfliers=False)
    for box_element in box1['boxes']:
        box_element.set(color='black', alpha=0.7)
        box_element.set_facecolor(color_list[i])

    # Plot data points
    for k in range(len(data[model_type])):
        ax.plot(i, data[model_type][k], 'o', markersize=4, color='red')

ax.set_xticks([i for i in range(len(models))])
ax.set_xticklabels(['RISE-iEEG', 'HTNet', 'EEGNet', 'Random Forest', 'Minimum Distance'], fontsize=14)

ax.set_ylabel('F1 score', fontsize=15)
# ax.set_title(f'Model version 2 (add {reg} regularization {reg_coef})', fontsize=14)
plt.ylim(0.5, 0.7)
fig.savefig(sp + task + '_' + balancing + '_' + compare_type + f'_fscore_{reg}_{int(reg_coef * 100)}_sns')
fig.savefig(sp + task + '_' + balancing + '_' + compare_type + f'_fscore_{reg}_{int(reg_coef * 100)}_sns.svg')
