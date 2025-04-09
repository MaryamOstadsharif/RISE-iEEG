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
    """
    Generate and save boxplots comparing F1-scores of multiple models across two tasks.

    Args:
        save_path (str): Directory to save the generated plots.
        path_save_comparison_model (dict): Paths to comparison models for each task.
        path_save_RISEiEEG_model (dict): Paths to RISE-iEEG models for each task.
    """
    num_fold = 10
    base_models = ['eegnet_hilb_', 'eegnet_', 'rf_', 'riemann_']
    all_models = ['RISEiEEG_'] + base_models
    tasks = ['Singing_Music', 'Move_Rest']

    # Load F1-scores for each model and task
    f1_score = {task: {} for task in tasks}
    for model_type in base_models:
        for task in tasks:
            file_path = os.path.join(path_save_comparison_model[task], f'fscore_gen_{model_type}{num_fold}.npy')
            f1_score[task][model_type] = np.load(file_path)[:, 2]

    for task in tasks:
        file_path = os.path.join(path_save_RISEiEEG_model[task], f'fscore_RISEiEEG_{num_fold}.npy')
        f1_score[task]['RISEiEEG_'] = np.load(file_path)[:, 2]

    # Plot settings
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), dpi=300)
    color_list = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
    pos_box = [0, 0.3, 0.6, 0.9, 1.2]
    model_labels = ['RISE-iEEG', 'HTNet', 'EEGNet', 'RF', 'MD']
    fig_titles = ['A)', 'B)']

    for t, task in enumerate(tasks):
        for i, model_type in enumerate(all_models):
            scores = f1_score[task][model_type]
            box = ax[t].boxplot(scores, patch_artist=True, positions=[pos_box[i]], widths=0.25, showfliers=False)

            # Styling boxplot
            for element in box['boxes']:
                element.set_edgecolor('black')
                element.set(color='black', alpha=0.7)
                element.set_facecolor(color_list[i])
            for median in box['medians']:
                median.set_color('black')
            for k in range(num_fold):
                ax[t].plot(pos_box[i], scores[k], 'o', markersize=3, color='black')

        # Axes styling
        ax[t].set_xticks(pos_box)
        ax[t].set_xticklabels(model_labels, fontsize=18, rotation=-45)
        ax[t].set_xlim(-0.2, 1.4)
        ax[t].spines['right'].set_visible(False)
        ax[t].spines['top'].set_visible(False)
        ax[t].set_ylabel('F1 score', fontsize=18)
        ax[t].tick_params(axis='y', which='major', labelsize=14)

        # Set y-axis limits/ticks based on task
        if task == 'Singing_Music':
            ax[t].set_yticks(np.arange(0.5, 0.95, 0.05))
            ax[t].set_ylim(0.5, 0.9)
        else:
            ax[t].set_yticks(np.arange(0.5, 0.8, 0.05))
            ax[t].set_ylim(0.49, 0.75)

        # Add subplot labels
        ax[t].annotate(fig_titles[t], xy=(-0.02, 1.1), xycoords='axes fraction',
                       fontsize=20, fontweight='bold', ha='center', va='center')

    # Save figure
    os.makedirs(save_path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'Comparison_same_patient.png'))
    fig.savefig(os.path.join(save_path, 'Comparison_same_patient.svg'))

    # Print mean ± std for each model and task
    for task in tasks:
        print(f'\nTask: {task}')
        for model_type in all_models:
            scores = f1_score[task][model_type]
            print(f' {model_type:<10}: {round(np.mean(scores), 2)} ± {round(np.std(scores), 2)}')


def scatter_plot_unseen_patient(save_path, path_save_comparison_model, path_save_RISEiEEG_model):
    """
    Generate scatter plots comparing model performance (F1-score) for each patient in the unseen patient setting.

    Args:
        save_path (str): Directory where the plots will be saved.
        path_save_comparison_model (dict): Dictionary containing paths to comparison model results.
        path_save_RISEiEEG_model (dict): Dictionary containing paths to RISE-iEEG results.
    """
    num_patient = {'Move_Rest': 12, 'Singing_Music': 29}
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

    for t, task in enumerate(['Move_Rest', 'Singing_Music']):
        n = num_patient[task]
        path_RISEiEEG = path_save_RISEiEEG_model[task]
        RISEiEEG_acc_mean = np.array([
            np.mean(np.load(os.path.join(path_RISEiEEG, f"{i}/fscore_ECoGNet_3.npy"))[:, 2])
            for i in range(n)
        ])

        # Order of patient indices for plotting
        ind_patient = np.array([
            5, 3, 1, 0, 4, 6, 10, 2, 9, 11, 8, 7
        ]) if task == 'Move_Rest' else np.array([
            4, 13, 0, 12, 5, 11, 10, 15, 3, 24, 17, 21, 19, 14, 2,
            16, 22, 1, 6, 9, 27, 20, 28, 18, 26, 25, 7, 8, 23
        ])

        # Initialize arrays for other models (HTNet, EEGNet, RF, Riemann)
        HTNet_acc = np.zeros((n, 3))
        EEGNet_acc = np.zeros((n, 3))
        RF_acc = np.zeros((n, 3))
        Riemann_acc = np.zeros((n, 3))

        # Load scores for each fold
        for k, path in enumerate(path_save_comparison_model[task]):
            HTNet_all = np.load(os.path.join(path, f"fscore_gen_eegnet_hilb_{n}.npy"))[:, 2]
            EEGNet_all = np.load(os.path.join(path, f"fscore_gen_eegnet_{n}.npy"))[:, 2]
            RF_all = np.load(os.path.join(path, f"fscore_gen_rf_{n}.npy"))[:, 2]
            Riemann_all = np.load(os.path.join(path, f"fscore_gen_riemann_{n}.npy"))[:, 2]

            for i in range(n):
                idx = np.where(ind_patient == i)[0][0]
                HTNet_acc[i, k] = HTNet_all[idx]
                EEGNet_acc[i, k] = EEGNet_all[idx]
                RF_acc[i, k] = RF_all[idx]
                Riemann_acc[i, k] = Riemann_all[idx]

        # Mean F1 scores
        HTNet_acc_mean = np.mean(HTNet_acc, axis=1)
        EEGNet_acc_mean = np.mean(EEGNet_acc, axis=1)
        RF_acc_mean = np.mean(RF_acc, axis=1)
        Riemann_acc_mean = np.mean(Riemann_acc, axis=1)

        # Print stats
        print(f'\nTask: {task}')
        print(f' RISE-iEEG        : {np.mean(RISEiEEG_acc_mean):.2f} ± {np.std(RISEiEEG_acc_mean):.2f}')
        print(f' HTNet            : {np.mean(HTNet_acc_mean):.2f} ± {np.std(HTNet_acc_mean):.2f}')
        print(f' EEGNet           : {np.mean(EEGNet_acc_mean):.2f} ± {np.std(EEGNet_acc_mean):.2f}')
        print(f' Random Forest    : {np.mean(RF_acc_mean):.2f} ± {np.std(RF_acc_mean):.2f}')
        print(f' Minimum Distance : {np.mean(Riemann_acc_mean):.2f} ± {np.std(Riemann_acc_mean):.2f}')

        # Plot
        ax[t].plot(RISEiEEG_acc_mean, 'D', label='RISE-iEEG', color='#E91E63', markersize=8)
        ax[t].plot(HTNet_acc_mean, 'o', label='HTNet', color='#3357FF', markersize=8)
        ax[t].plot(EEGNet_acc_mean, 's', label='EEGNet', color='cyan', markersize=8)
        ax[t].plot(RF_acc_mean, '^', label='Random Forest', color='#8E44AD', markersize=8)
        ax[t].plot(Riemann_acc_mean, 'p', label='Minimum Distance', color='limegreen', markersize=8)

        ax[t].spines['right'].set_visible(False)
        ax[t].spines['top'].set_visible(False)
        ax[t].set_xticks(np.arange(n))
        ax[t].set_xticklabels([f'P{i+1}' for i in range(n)], fontsize=10, rotation=-45)
        ax[t].set_yticks(np.arange(0.4, 1.2, 0.1))
        ax[t].set_ylim(0.4, 1.2)
        ax[t].set_ylabel('F1 score', fontsize=15)
        ax[t].legend(ncol=2, fontsize=12)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, "comparison_models_sing_music_scatter_point.png"))
    fig.savefig(os.path.join(save_path, "comparison_models_sing_music_scatter_point.svg"))

def plot_elec_point(patient, path_coordinates, path_save, mark_size=10):
    """
    Generate and save a 3D brain visualization of electrode locations for a specific patient.

    Args:
        patient (int): Index of the patient.
        path_coordinates (str): Path to the .pkl file containing electrode coordinates.
        path_save (str): Directory where the HTML plot should be saved.
        mark_size (int): Size of the electrode markers.
    """
    with open(path_coordinates, 'rb') as f:
        coor_elec = pkl.load(f)[patient]

    plot = plotting.view_markers(coor_elec, marker_size=mark_size)
    os.makedirs(path_save, exist_ok=True)
    plot.save_as_html(os.path.join(path_save, f'plot_elec_point_patient{patient}.html'))

def plot_heatmap_ig(patient_id, trial_id, save_path, ig_path, ch_name_path):
    """
    Generate and save heatmaps of Integrated Gradients (IG) for a given patient and trial.

    Args:
        patient_id (int): Index of the patient.
        trial_id (list): Indices defining the event trial ranges (start, end).
        save_path (str): Directory to save the generated heatmaps.
        ig_path (str): Path to the Integrated Gradients (IG) data.
        ch_name_path (str): Path to the channel names.
    """
    # Load channel names and IG data
    with open(ch_name_path, 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(ig_path)

    fig, ax = plt.subplots(1, 2, dpi=300)
    event_list = ['Music', 'Singing']

    for i in range(2):
        # Normalize the absolute mean IG data
        data_norm = preprocessing.normalize(np.abs(np.mean(ig[patient_id][trial_id[i]:trial_id[i + 1], :, :], axis=0)))

        # Create heatmap
        im = ax[i].imshow(data_norm[::-1, :len(ch[patient_id])], cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax[i].set_title(f'{event_list[i]} event')

        # Add colorbar
        divider = make_axes_locatable(ax[i])
        cax_cb = divider.append_axes("right", size="7%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax_cb)
        cbar.ax.tick_params(labelsize=8)

        # Set ticks and labels
        ax[i].set_yticks(np.arange(0, 225, 25))
        ax[i].set_xticks(np.arange(0, len(ch[patient_id]) + 8, 8))
        ax[i].set_yticklabels(np.arange(200, -25, -25), fontsize=8)
        ax[i].set_xticklabels(np.arange(0, len(ch[patient_id]) + 8, 8), fontsize=8)
        ax[i].set_ylabel('Time')
        ax[i].set_xlabel('Channel')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure
    fig.savefig(f'{save_path}/Heatmap_IG_{patient_id}.png')
    fig.savefig(f'{save_path}/Heatmap_IG_{patient_id}.svg')

def plot_markers_ig(patient, trial_id, step_time, task, ig_path, elec_coor_path, hemisphere_path, save_path):
    """
    Plot markers on a brain model for a given patient, task, and event based on the Integrated Gradients (IG).

    Args:
        patient (int): Index of the patient.
        trial_id (list): Indices defining the event trial ranges (start, end).
        step_time (int): Time step for the plots in milliseconds.
        task (str): The task type ('Singing_Music' or 'Move_Rest').
        ig_path (str): Path to the Integrated Gradients (IG) data.
        elec_coor_path (str): Path to the electrode coordinates.
        hemisphere_path (str): Path to the hemisphere information (left or right).
        save_path (str): Directory to save the generated plots.
    """
    # Load data
    ig = np.load(ig_path)
    with open(elec_coor_path, 'rb') as f:
        elec_coor = pkl.load(f)
    hem = np.load(hemisphere_path)

    # Event titles and sampling frequency based on the task
    title_list = ['Music event', 'Singing event'] if task == 'Singing_Music' else ['Rest event', 'Move event']
    fs = 100 if task == 'Singing_Music' else 250

    # Number of rows for plotting
    n_row = 2
    T = int((step_time / 1000) * fs)  # Convert step_time to samples
    fig, ax = plt.subplots(n_row, int(ig[patient][0, :, :].shape[0] / T) + 1, figsize=(27, 8), dpi=300)

    # Plot the data for each event
    for i in range(n_row):
        for time in range(int(ig[patient][0, :, :].shape[0] / T) + 1):
            # Normalize and plot data for each trial
            data_in = preprocessing.normalize(np.mean(ig[patient][trial_id[i]:trial_id[i + 1], :, :], axis=0)).T[:len(elec_coor[patient]), time * T - 1]
            plot = plotting.plot_markers(data_in, elec_coor[patient], node_size=80, node_cmap='bwr',
                                         alpha=1, output_file=None, display_mode=hem[patient], axes=ax[i, time],
                                         annotate=True, black_bg=False, colorbar=True, radiological=False, node_vmin=0)

            # Adjust colorbar font size
            cbar = plot._cbar
            cbar.ax.tick_params(labelsize=20)

            # Set titles for each subplot with time
            title = ax[0, time].set_title(f'T= {time * T / fs:g} sec', fontsize=30, pad=25, weight='bold')
            title.set_position((title.get_position()[0] - 0.06, title.get_position()[1]))

        # Add event title for each row
        fig.text(0, 0.7 - 0.42 * i, title_list[i], fontsize=30, ha='left', weight='bold')

    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f'{save_path}/{task}_markers_heatmap_patient_{patient + 1}_move_rest.png')
    fig.savefig(f'{save_path}/{task}_markers_heatmap_patient_{patient + 1}_move_rest.svg')


def plot_histogram(num_best_elec, task, ig_path, channel_name_path, save_path, num_lobes, plot_curve_best_lobes):
    """
    Plot a histogram of significant lobes and optionally plot curves of best lobes for each participant.

    Args:
        num_best_elec (int): Number of best electrodes to consider.
        task (str): The task ('Singing_Music' or other).
        ig_path (str): Path to the Integrated Gradients (IG) data.
        channel_name_path (str): Path to the channel names.
        save_path (str): Directory to save the generated plots.
        num_lobes (int): Number of lobes to plot.
        plot_curve_best_lobes (bool): Whether to plot the curve for the best lobes.
    """
    # Load channel names and IG data
    with open(channel_name_path, 'rb') as f:
        ch_names = pkl.load(f)
    ig = np.load(ig_path)

    # Initialize arrays to store best electrode positions and labels
    pos_elec_best = np.zeros((len(ig), len(ig[0])))
    label_elec_best = np.zeros((len(ig), len(ig[0])), dtype='U30')
    count_label = []

    # Process data for each patient and event
    for patient in range(len(ig)):
        for event in range(len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event] = pos
            if task == 'Singing_Music':
                label_elec_best[patient, event] = ch_names[patient][pos][7:]  # Exclude prefix
            else:
                label_elec_best[patient, event] = ch_names[patient][pos]
        count_label.extend(np.array(Counter(label_elec_best[patient, :]).most_common())[0:num_best_elec, 0].tolist())

    # Plot the curves of best lobes if required
    if plot_curve_best_lobes:
        # Create directories to save the curves
        curves_path = os.path.join(save_path, f'Curves_best_lobes_{task}')
        os.makedirs(curves_path, exist_ok=True)

        num_participants = len(ig)
        cols = 6  # Number of columns in the subplot grid
        rows = (num_participants // cols) + (num_participants % cols > 0)  # Calculate rows

        # Create the figure for subplots
        plt.figure(dpi=300, figsize=(15, rows * 4))

        # Plot for each participant
        for patient in range(num_participants):
            count_dict = Counter(label_elec_best[patient, :]).most_common()
            count = [count_dict[j][1] for j in range(len(count_dict))]
            count_norm = (count - np.min(count)) / (np.max(count) - np.min(count))
            count_norm_ = [round(count_norm[k], 4) * 100 for k in range(len(count_norm))]

            # Create subplot for each participant
            ax = plt.subplot(rows, cols, patient + 1)
            ax.plot(np.arange(1, len(count_norm_) + 1), count_norm_, linewidth=2)

            # Detect the 'knee' point in the curve
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

        # Save the subplot figure
        plt.tight_layout()
        plt.savefig(os.path.join(curves_path, 'Curves_best_lobes_subplot.png'))
        plt.savefig(os.path.join(curves_path, 'Curves_best_lobes_subplot.svg'))

    # Print out the count of labels
    print(Counter(count_label))

    # Plot the histogram of most common lobes
    fig, ax = plt.subplots(dpi=300)
    color_list = ['steelblue', 'deeppink', 'cyan', 'purple', 'lime', 'gold', 'crimson', 'darkgray', 'sienna',
                  'forestgreen', 'navy', 'forestgreen', 'blue', 'violet', 'red', 'brown', 'green', 'gray', 'darkblue']

    for i in range(num_lobes):
        # Plot bars for the most common lobes
        plt.bar(i, (Counter(count_label).most_common()[i][1] / len(ch_names)) * 100, color=color_list[i], ecolor='blue',
                capsize=3, label=Counter(count_label).most_common()[i][0])

    # Customize the appearance of the plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if task == 'Singing_Music':
        dict_reg = {
            'superiortemporal': 'ST',
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
            'inferiorparietal': 'IP'
        }
        plt.xticks(range(num_lobes), [dict_reg[Counter(count_label).most_common()[i][0]] for i in range(num_lobes)],
                   fontsize=12)
        plt.ylim(0, 90)
        plt.yticks(np.arange(0, 100, 10), fontsize=12)
    else:
        dict_reg = {
            'Postcentral Gyrus': 'PC',
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
            'Occipital Pole': 'O'
        }
        plt.xticks(range(num_lobes), [dict_reg[Counter(count_label).most_common()[i][0]] for i in range(num_lobes)],
                   fontsize=12)
        plt.ylim(0, 60)
        plt.yticks(np.arange(0, 70, 10), fontsize=12)

    # Final plot settings
    plt.ylabel('Percentage of participants', fontsize=12)
    plt.xlabel('Significant lobes', fontsize=12)
    plt.tight_layout()

    # Save the histogram plot
    plt.savefig(os.path.join(save_path, f'hist_29p_{task}.png'))
    plt.savefig(os.path.join(save_path, f'hist_29p_{task}.svg'))


def plot_brain_surface(task, path_save):
    """
    Visualize brain surface with lobe intensities based on a given task.

    Args:
        task (str): The task for which the brain surface is plotted (e.g., 'Move_Rest').
        path_save (str): Path to save the generated HTML visualization of the brain surface.
    """
    # Load the Destrieux atlas and fsaverage surface from nilearn datasets
    destrieux = datasets.fetch_atlas_surf_destrieux()
    fsaverage = datasets.fetch_surf_fsaverage()

    # Initialize surface values for the left hemisphere (zeroed initially)
    surface_values = np.zeros(destrieux['map_left'].shape)

    # Define lobe intensity mappings for different tasks
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

    # Update surface values based on the defined lobe intensity map
    for lobe, intensity in lobe_intensity_map.items():
        # Find indices corresponding to the lobe labels
        lobe_indices = [i for i, label in enumerate(destrieux['labels']) if bytes(lobe, 'utf-8') in label]
        lobe_voxels = np.concatenate([np.where(destrieux['map_left'] == i)[0] for i in lobe_indices])

        # Assign intensity to the surface values at the corresponding voxel positions
        surface_values[lobe_voxels] = intensity

    # Visualize the surface with the updated intensities
    plot = plotting.view_surf(fsaverage['infl_left'], surface_values, cmap='Blues', symmetric_cmap=False)

    # Save the plot as an interactive HTML file
    plot.save_as_html(f"{path_save}plot_brain_surface_{task}.html")


def get_best_lobes_weight_denselayers(path_save_model, channel_name_path, num_participants):
    """
    Extract the best lobe based on the highest summed weights in the dense layer of a model for each participant.

    Args:
        path_save_model (str): Path to the saved model.
        channel_name_path (str): Path to the pickle file containing channel names.
        num_participants (int): The number of participants to process.

    Returns:
        pd.DataFrame: A DataFrame containing the best lobe for each participant.
    """
    # Load channel names and pre-trained model
    with open(channel_name_path, 'rb') as f:
        all_ch = pkl.load(f)

    pretrained_model = tf.keras.models.load_model(path_save_model)

    # Initialize a list to store the best lobe for each participant
    best_lobe_all_participants = []

    # Process each participant to find their best lobe based on model weights
    for participant in range(num_participants):
        # Get the dense layer weights specific to the current participant
        dense_weight = pretrained_model.layers[num_participants + participant].get_weights()[0]

        # Find the index of the lobe with the highest summed weight
        pos_best = np.argmax(np.sum(dense_weight, axis=1))

        # Get the name of the best lobe for the current participant
        best_lobe = all_ch[participant][pos_best]

        # Store the result for the current participant
        best_lobe_all_participants.append({'Participant': participant, 'Best Lobe': best_lobe})

    # Convert the result to a pandas DataFrame for easy tabular format
    best_lobe_df = pd.DataFrame(best_lobe_all_participants)

    return best_lobe_df
