import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle as pkl
from collections import Counter
from nilearn import plotting, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kneed import KneeLocator
import pandas as pd
import tensorflow as tf

def plot_box_compare_models(save_path, path_save_comparison_model, path_save_RISEiEEG_model):
    # Number of folds
    num_fold = 10
    # List of model types to compare
    models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']
    # Dictionary to store the data for each model
    f1_score = {'Singing_Music': {}, 'Move_Rest': {}}

    for model_type in models:
        for task in f1_score.keys():
            f1_score[task][model_type] = np.load(
                path_save_comparison_model[task] + '/fscore_gen_' + model_type + str(num_fold) + '.npy')[:, 2]

    models = ['RISEiEEG_'] + models
    for task in f1_score.keys():
        f1_score[task][models[0]] = np.load(
            path_save_RISEiEEG_model[task] + '/fscore_' + models[0] + str(num_fold) + '.npy')[:, 2]


    # Create a figure and axis for the plot
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), dpi=300)
    # List of colors for the box plots
    color_list = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
    pos_box = [0, 0.3, 0.6, 0.9, 1.2]
    text_fig = ['A)', 'B)']
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

    for task in f1_score.keys():
        print(f'\n Task: {task}')
        for model_type in models:
            print(F' {model_type} : {round(np.mean(f1_score[task][model_type]), 2)} ± '
                  F'{round(np.std(f1_score[task][model_type]), 2)}')

def scatter_plot_unseen_patient(save_path, path_save_comparison_model, path_save_RISEiEEG_model):
    num_patient = {'Move_Rest': 12, 'Singing_Music': 29}

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

    for t, task in enumerate(['Move_Rest', 'Singing_Music']):
        RISEiEEG_acc_mean = np.zeros(num_patient[task])
        path_RISEiEEG = path_save_RISEiEEG_model[task]
        for i in range(num_patient[task]):
            RISEiEEG_acc_mean[i] = np.mean(np.load(path_RISEiEEG + str(i) + "/fscore_ECoGNet_3.npy")[:, 2])

        paths_comp_models = path_save_comparison_model[task]

        if task == 'Singing_Music':
            ind_patient = np.array([4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2,
                                    16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23])
        else:
            ind_patient = np.array([5, 3, 1, 0, 4, 6, 10, 2, 9, 11, 8, 7])

        HTNet_acc = np.zeros((num_patient[task], 3))
        EEGNet_acc = np.zeros((num_patient[task], 3))
        RF_acc = np.zeros((num_patient[task], 3))
        Rieman_acc = np.zeros((num_patient[task], 3))
        for k, path in enumerate(paths_comp_models):
            HTNet_acc_all = np.load(path + f"fscore_gen_eegnet_hilb_{num_patient[task]}.npy")[:, 2]
            EEGNet_acc_all = np.load(path + f"fscore_gen_eegnet_{num_patient[task]}.npy")[:, 2]
            RF_acc_all = np.load(path + f"fscore_gen_rf_{num_patient[task]}.npy")[:, 2]
            Rieman_acc_all = np.load(path + f"fscore_gen_riemann_{num_patient[task]}.npy")[:, 2]

            for i in range(num_patient[task]):
                HTNet_acc[i, k] = HTNet_acc_all[np.where(ind_patient == i)]
                EEGNet_acc[i, k] = EEGNet_acc_all[np.where(ind_patient == i)]
                RF_acc[i, k] = RF_acc_all[np.where(ind_patient == i)]
                Rieman_acc[i, k] = Rieman_acc_all[np.where(ind_patient == i)]

        HTNet_acc_mean = np.mean(HTNet_acc, axis=1)
        EEGNet_acc_mean = np.mean(EEGNet_acc, axis=1)
        RF_acc_mean = np.mean(RF_acc, axis=1)
        Rieman_acc_mean = np.mean(Rieman_acc, axis=1)

        print(f'\n Task: {task}')
        print(
            F' RISE-iEEG : {round(np.mean(RISEiEEG_acc_mean), 2)} ± {round(np.std(RISEiEEG_acc_mean), 2)}')
        print(F' HTNet : {round(np.mean(HTNet_acc_mean ), 2)} ± {round(np.std(HTNet_acc_mean), 2)}')
        print(F' EEGNet : {round(np.mean(EEGNet_acc_mean ), 2)} ± {round(np.std(EEGNet_acc_mean), 2)}')
        print(F' Random Forest : {round(np.mean(RF_acc_mean), 2)} ± {round(np.std(RF_acc_mean), 2)}')
        print(
            F' Minimum Distance : {round(np.mean(Rieman_acc_mean), 2)} ± {round(np.std(Rieman_acc_mean), 2)}')

        # Plotting the data
        ax[t].plot(RISEiEEG_acc_mean, 'D', label='RISE-iEEG', color='#E91E63', markersize=8)
        ax[t].plot(HTNet_acc_mean, 'o', label='HTNet', color='#3357FF', markersize=8)
        ax[t].plot(EEGNet_acc_mean, 's', label='EEGNet', color='cyan', markersize=8)
        ax[t].plot(RF_acc_mean, '^', label='Random Forest', color='#8E44AD', markersize=8)
        ax[t].plot(Rieman_acc_mean, 'p', label='Minimum Distance', color='limegreen', markersize=8)

        # Hide the top and right spines
        ax[t].spines['right'].set_visible(False)
        ax[t].spines['top'].set_visible(False)

        # Set x-ticks and labels
        ax[t].set_xticks(np.arange(num_patient[task]))
        ax[t].set_xticklabels([f'P{i + 1}' for i in range(num_patient[task])], fontsize=10, rotation=-45)

        # Set y-ticks and range
        ax[t].set_yticks(np.arange(0.4, 1.2, 0.1))
        ax[t].set_ylim(0.4, 1.2)

        # Axis labels
        ax[t].set_ylabel('F1 score', fontsize=15)

        # Add legend
        ax[t].legend(ncol=2, fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path + "/comparison_models_sing_music_scatter_point.png")
    fig.savefig(save_path + "/comparison_models_sing_music_scatter_point.svg")


def plot_elec_point(patient, path_coordinates, path_save, mark_size=10):
    with open(path_coordinates, 'rb') as f:
        coor_elec = pkl.load(f)[patient]
    plot = plotting.view_markers(coor_elec, marker_size=mark_size)
    plot.save_as_html(path_save + f'plot_elec_point_patient{patient}.html')


def plot_heatmap_ig(patient_id, trial_id, save_path, ig_path, ch_name_path):
    with open(ch_name_path, 'rb') as f:
        ch = pkl.load(f)

    ig = np.load(ig_path)

    fig, ax = plt.subplots(1, 2, dpi=300)
    Event_list = ['Music', 'Singing']

    for i in range(2):
        data_norm = preprocessing.normalize(np.abs(np.mean(ig[patient_id][trial_id[i]:trial_id[i + 1], :, :], axis=0)))

        im = ax[i].imshow(data_norm[::-1, :len(ch[patient_id])], cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax[i].set_title(f'{Event_list[i]} event')
        divider = make_axes_locatable(ax[i])
        cax_cb = divider.append_axes("right", size="7%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax_cb)
        cbar.ax.tick_params(labelsize=8)
        ax[i].set_yticks(np.arange(0, 225, 25))
        ax[i].set_xticks(np.arange(0, len(ch[patient_id]) + 8, 8))
        ax[i].set_yticklabels(np.arange(200, -25, -25), fontsize=8)
        ax[i].set_xticklabels(np.arange(0, len(ch[patient_id]) + 8, 8), fontsize=8)
        ax[i].set_ylabel('Time')
        ax[i].set_xlabel('Channel')

    plt.tight_layout()
    fig.savefig(save_path + f'/Heatmap_IG_{patient_id}.png')
    fig.savefig(save_path + f'/Heatmap_IG_{patient_id}.svg')


def plot_markers_ig(patient, trial_id, step_tiem, task, ig_path, elec_coor_path, hemisphere_path, save_path):
    ig = np.load(ig_path)
    with open(elec_coor_path, 'rb') as f:
        elec_coor = pkl.load(f)
    hem = np.load(hemisphere_path)

    if task == 'Singing_Music':
        title_list = ['Music event', 'Singing event']
        fs = 100
    else:
        title_list = ['Rest event', 'Move event']
        fs = 250

    n_row = 2
    T = int((step_tiem / 1000) * fs)
    fig, ax = plt.subplots(n_row, int(ig[patient][0, :, :].shape[0] / T)+1, figsize=(27, 8), dpi=300)

    for i in range(n_row):
        for time in range(int(ig[patient][0, :, :].shape[0] / T) + 1):
            data_in = preprocessing.normalize(
                np.mean(ig[patient][trial_id[i]:trial_id[i + 1], :, :], axis=0)).T[:len(elec_coor[patient]),
                      time * T - 1]
            plot = plotting.plot_markers(data_in, elec_coor[patient], node_size=80, node_cmap='bwr',
                                         alpha=1, output_file=None, display_mode=hem[patient], axes=ax[i, time],
                                         annotate=True, black_bg=False, colorbar=True, radiological=False, node_vmin=0)

            # Adjust the colorbar font size
            cbar = plot._cbar
            cbar.ax.tick_params(labelsize=20)

            title = ax[0, time].set_title(f'T= {time * T / fs:g} sec', fontsize=30, pad=25, weight='bold')
            title.set_position((title.get_position()[0] - 0.06, title.get_position()[1]))

        fig.text(0, 0.7 - 0.42 * i, title_list[i], fontsize=30, ha='left', weight='bold')

    # fig.tight_layout()
    fig.savefig(save_path + f'{task}_markers_heatmap_patient_{patient + 1}_move_rest.png')
    fig.savefig(save_path + f'{task}_markers_heatmap_patient_{patient + 1}_move_rest.svg')


def plot_histogram(num_best_elec, task, ig_path, channel_name_path, save_path, num_lobes, plot_curve_best_lobes):
    with open(channel_name_path, 'rb') as f:
        ch_names = pkl.load(f)
    ig = np.load(ig_path)

    pos_elec_best = np.zeros((len(ig), len(ig[0])))
    label_elec_best = np.zeros((len(ig), len(ig[0])), dtype='U30')
    count_label = []
    for patient in range(len(ig)):
        for event in range(len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event] = pos
            if task == 'Singing_Music':
                label_elec_best[patient, event] = ch_names[patient][pos][7:]
            else:
                label_elec_best[patient, event] = ch_names[patient][pos]
        count_label.extend(np.array(Counter(label_elec_best[patient, :]).most_common())[0:num_best_elec, 0].tolist())

    if plot_curve_best_lobes:
        if not os.path.exists(save_path + f'/Curves_best_lobes_{task}'):
            os.makedirs(save_path + f'/Curves_best_lobes_{task}')

        num_participants = len(ig)
        cols = 6  # Number of columns in the subplot grid
        rows = (num_participants // cols) + (num_participants % cols > 0)  # Calculate rows needed

        plt.figure(dpi=300, figsize=(15, rows * 4))  # Adjust figure size based on rows

        for patient in range(num_participants):
            count_dict = Counter(label_elec_best[patient, :]).most_common()
            count = [count_dict[j][1] for j in range(len(count_dict))]
            count_norm = (count - np.min(count)) / (np.max(count) - np.min(count))
            count_norm_ = [round(count_norm[k], 4) * 100 for k in range(len(count_norm))]

            # Create subplot for each participant
            ax = plt.subplot(rows, cols, patient + 1)
            ax.plot(np.arange(1, len(count_norm_) + 1), count_norm_, linewidth=2)

            knee = KneeLocator(np.arange(len(count_norm_)), count_norm_, curve='convex', direction='decreasing')
            if knee.knee is not None and knee.knee_y is not None:
                ax.plot(knee.knee + 1, knee.knee_y, 'r*', markersize=10)
            else:
                print(f"No elbow point detected for participant {patient}.")
                if patient == 26:
                    ax.plot(3, count_norm_[2], 'r*', markersize=10)
                if patient == 28:
                    ax.plot(2, count_norm_[1], 'r*', markersize=10)

            ax.set_title(f'Participant {patient + 1}', fontsize=15, fontweight='bold')
            ax.set_xticks(np.arange(1, len(count_norm_) + 1))
            ax.set_ylabel('Percentage of IG information', fontsize=12)
            ax.set_xlabel('Number of lobes', fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path + f'/Curves_best_lobes_{task}/Curves_best_lobes_subplot.png')
        plt.savefig(save_path + f'/Curves_best_lobes_{task}/Curves_best_lobes_subplot.svg')

    print(Counter(count_label))

    fig, ax = plt.subplots(dpi=300)
    color_list = ['steelblue', 'deeppink', 'cyan', 'purple', 'lime', 'gold', 'crimson', 'darkgray', 'sienna',
                  'forestgreen', 'navy', 'forestgreen', 'blue', 'violet', 'red', 'brown', 'green', 'gray', 'darkblue']

    for i in range(num_lobes):
        plt.bar(i, (Counter(count_label).most_common()[i][1] / len(ch_names)) * 100, color=color_list[i], ecolor='blue',
                capsize=3, label=Counter(count_label).most_common()[i][0])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if task == 'Singing_Music':
        dict_reg = {'superiortemporal': 'ST',
                    'rostralmiddlefrontal': 'RMF',
                    'middletemporal': 'MT',
                    'postcentral': 'PC',
                    'precentral': 'PrC',
                    'caudalmiddlefrontal': 'CMF',
                    'parsopercularis': 'PO',
                    'inferiortemporal': 'IT',
                    'supramarginal': 'SM',
                    'superiorfrontal': 'SF',
                    'lateraloccipital': 'LO',
                    'parstriangularis': 'PT',
                    'inferiorparietal': 'IP'}
        plt.xticks(range(num_lobes), [dict_reg[Counter(count_label).most_common()[i][0]] for i in range(num_lobes)],
                   fontsize=12)
        plt.ylim(0, 90)
        plt.yticks(np.arange(0, 100, 10), fontsize=12)
    else:
        dict_reg = {'Postcentral Gyrus': 'PC',
                    'Occipital Fusiform Gyrus': 'OF',
                    'Superior Parietal Lobule': 'SP',
                    'Temporal Fusiform Cortex': 'TF',
                    'Temporal Pole': 'T',
                    'Precentral Gyrus': 'PrC',
                    'Lateral Occipital Cortex': 'LO',
                    'Central Opercular Cortex': 'CO',
                    'Middle Temporal Gyrus': 'MT',
                    'Supramarginal Gyrus': 'SM',
                    'Middle Frontal Gyrus': 'MF',
                    'Temporal Occipital Fusiform Co': 'TOF',
                    'Occipital Pole': 'O'}
        plt.xticks(range(num_lobes), [dict_reg[Counter(count_label).most_common()[i][0]] for i in range(num_lobes)],
                   fontsize=12)
        plt.ylim(0, 60)
        plt.yticks(np.arange(0, 70, 10), fontsize=12)
    plt.ylabel('Percentage of participants', fontsize=12)
    plt.xlabel('Significant lobes', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path + f'hist_29p_{task}.png')
    plt.savefig(save_path + f'hist_29p_{task}.svg')


def plot_brain_surface(task, path_save):
    # Load the Destrieux atlas and the fsaverage surface
    destrieux = datasets.fetch_atlas_surf_destrieux()
    fsaverage = datasets.fetch_surf_fsaverage()

    # Initialize a zero array for the left hemisphere surface map
    surface_values = np.zeros(destrieux['map_left'].shape)

    # Define lobes and their corresponding intensity values
    if task == 'Move_Rest':
        lobe_intensity_map = {
            'postcentral': 4,
            'precentral': 2,
            'temporal_inf': 1,
            'Pole_temporal': 2,
            'parietal_sup': 1,
            'Supramar': 1,
            'temporal_sup': 1}
    else:
        lobe_intensity_map = {
          'precentral': 3,
          'temporal_inf': 1,
          'front_sup': 1,
          'Supramar': 1,
          'temporal_sup': 21,
          'front_middle': 1,
          'G_orbital': 1,
          'temp_sup': 21}


    # Update surface values based on lobe labels
    for lobe, intensity in lobe_intensity_map.items():
      lobe_indices = [i for i, label in enumerate(destrieux['labels']) if bytes(lobe, 'utf-8') in label]
      lobe_voxels = np.concatenate([np.where(destrieux['map_left'] == i)[0] for i in lobe_indices])
      surface_values[lobe_voxels] = intensity

    # Visualize the surface
    plot = plotting.view_surf(fsaverage['infl_left'], surface_values, cmap='Blues', symmetric_cmap=False)
    plot.save_as_html(path_save + f'plot_brain_surface_{task}.html')

def get_best_lobes_weight_denselayers(path_save_model, channel_name_path, num_participants):
    with open(channel_name_path, 'rb') as f:
        all_ch = pkl.load(f)

    # Load the pre-trained model
    pretrained_model = tf.keras.models.load_model(path_save_model)

    # Initialize a list to store the best lobe for each participant
    best_lobe_all_participants = []

    # Loop through each participant to find their best lobe
    for participant in range(num_participants):
        # Extract the weights of the dense layer specific to each participant
        dense_weight = pretrained_model.layers[num_participants + participant].get_weights()[0]

        # Find the position of the lobe with the highest summed weight
        pos_best = np.argmax(np.sum(dense_weight, axis=1))

        # Get the best lobe and store it in the list (slicing for the lobe name)
        best_lobe = all_ch[participant][pos_best]
        best_lobe_all_participants.append({'Participant': participant, 'Best Lobe': best_lobe})

    # Convert the list to a DataFrame for a table format
    best_lobe_df = pd.DataFrame(best_lobe_all_participants)

    return best_lobe_df
