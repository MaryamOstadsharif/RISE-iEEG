import matplotlib.pyplot as plt
import numpy as np

# save path
sp = "F:/maryam_sh/General model/plots/comparison_models_"
# 'Question_Answer' & 'Singing_Music' & 'Speech_Music'
task = 'Singing_Music'
# number of folds for dataset: 'audio_visual' & 'under_sampling'=20 ,
# 'audio_visual' & 'under_sampling'=10,
# for dataset: 'music_reconstruction':29
num_fold = 29
# 'under_sampling' or 'over_sampling'
balancing = 'over_sampling'

models = []
data = {}
if task == 'Singing_Music':
    models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']
    for model_type in models:
        data[model_type] = []
        data_one = np.load(
            'F:/maryam_sh/Htnet_mydata_2/results/' + task + '/' + balancing + '/2023-12-05-23-47-29/accuracy/combined_sbjs_power'
                                                                              '/acc_gen_' + model_type + str(
                num_fold) + '.npy')
        data[model_type] = data_one[:, 2]

models = ['GNCNN_'] + models
data_one = np.load(
    'F:/maryam_sh/General model/General code/results/' + task + '/' + balancing + '/2024-01-27-13-30-18/accuracy'
                                                                                  '/acc_gen_' + models[0] + str(
        num_fold) + '.npy')
data[models[0]] = data_one[:, 2]

# Create a boxplot for the first group
fig, ax = plt.subplots(figsize=(8,6), dpi=300)
color_list = ['yellow', 'cyan', 'lime', 'blue', 'purple', 'darkgray', 'pink', 'darkgray', 'orangered', 'deeppink']

for i, model_type in enumerate(models):
    box1 = ax.boxplot(data[model_type], patch_artist=True, positions=[i], widths=0.3,
                      showfliers=False)  # Adjust the width and position as needed
    for box_element in box1['boxes']:
        box_element.set(color='black', alpha=0.7)
        box_element.set_facecolor(color_list[i])  # Set the facecolor

    for k in range(len(data[model_type])):
        ax.plot(i, data[model_type][k], 'o', markersize=3, color='red')

# Customize the plot
ax.set_xticks([i for i in range(len(models))])
if task == 'Singing_Music':
    ax.set_xticklabels(['GN_CNN', 'HTNet', 'EEGNet', 'RandomForest', 'MinimumDistance'])
else:
    ax.set_xticklabels(['GN_CNN'])
ax.set_ylabel('Test accuracy')
ax.set_title(('Multi patients models(' + task + '_' + balancing + ') in ' + str(num_fold) + ' folds'), fontsize=12)

# Show the plot
# plt.show()
fig.savefig(sp + task + '_' + balancing)
