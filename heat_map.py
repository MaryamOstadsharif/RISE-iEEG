import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle as pkl
from collections import Counter
from nilearn import plotting


def plot_elec_point(patient):
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    coor_elec = node_coords[patient]
    plot = plotting.view_markers(coor_elec, marker_size=10)
    plot.save_as_html(f'plot_elec_point_patient{patient}.html')


def plot_heatmap_per_patient(patient_id, trial_id):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)

    ig = np.load('F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')

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
        ax[i].set_xticks(np.arange(0, len(ch[patient_id])+8, 8))
        ax[i].set_yticklabels(np.arange(200, -25, -25), fontsize=8)
        ax[i].set_xticklabels(np.arange(0, len(ch[patient_id]) + 8, 8), fontsize=8)
        ax[i].set_ylabel('Time')
        ax[i].set_xlabel('Channel')

    # ch[patient_id][np.argmax(np.sum(data_norm, axis=0))
    plt.tight_layout()
    fig.savefig(f'F:/maryam_sh/integrated_grad/ht_{patient_id}.png')
    fig.savefig(f'F:/maryam_sh/integrated_grad/ht_{patient_id}.svg')

def plot_histogram(event_id_f, num_best_elec):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)

    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')

    pos_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f))
    label_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f), dtype='U30')
    count_label = []
    for patient in range(len(ig)):
        for event in range(event_id_f, len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event - event_id_f] = pos
            label_elec_best[patient, event - event_id_f] = ch[patient][pos][7:]
        count_label.extend(np.array(Counter(label_elec_best[patient, :]).most_common())[0:num_best_elec, 0].tolist())

    print(Counter(count_label))
    fig, ax = plt.subplots(dpi=300)
    color_list = ['yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'violet', 'darkgray', 'orangered',
                  'forestgreen', 'navy', 'forestgreen', 'blue', 'violet', 'red', 'brown', 'green', 'gray', 'darkblue']

    for i in range(10):
        plt.bar(i, Counter(count_label).most_common()[i][1], color=color_list[i], ecolor='blue', capsize=3,
                label=Counter(count_label).most_common()[i][0])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.xticks([])
    plt.ylim(0, Counter(count_label).most_common()[0][1] + 2)
    plt.ylabel('#Patients')
    plt.xlabel('Valuable lobes')
    # plt.title(f'Histogram significant electrodes across 29 patients for {len(ig[0]) - event_id_f} events', fontsize=9)
    plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p.png')
    plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p.svg')
    print('end')


def plot_hist_seperate(num_best_elec):
    event_id_f = 0
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')

    pos_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f))
    label_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f), dtype='U30')

    fig, ax = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

    count_label_music = []
    count_label_singing = []
    for patient in range(len(ig)):
        for event in range(event_id_f, len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event - event_id_f] = pos
            label_elec_best[patient, event - event_id_f] = ch[patient][pos][7:]
        count_label_singing.extend(
            np.array(Counter(label_elec_best[patient, :16]).most_common())[0:num_best_elec, 0].tolist())
        count_label_music.extend(
            np.array(Counter(label_elec_best[patient, 16:]).most_common())[0:num_best_elec, 0].tolist())

    color_list = ['yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'pink', 'darkgray', 'orangered',
                  'forestgreen', 'navy', 'forestgreen']

    for i in range(len(Counter(count_label_singing))):
        ax[0].bar(i, Counter(count_label_singing).most_common()[i][1], color=color_list[i], ecolor='blue', capsize=3,
                  label=Counter(count_label_singing).most_common()[i][0])
    for i in range(len(Counter(count_label_music))):
        ax[1].bar(i, Counter(count_label_music).most_common()[i][1], color=color_list[i], ecolor='blue', capsize=3,
                  label=Counter(count_label_music).most_common()[i][0])

    ax[0].legend(fontsize=11)
    ax[0].set_title('Singing trial', fontsize=18)

    ax[1].legend(fontsize=11)
    ax[1].set_title('Music trial', fontsize=18)

    plt.ylim(0, Counter(count_label_music).most_common()[0][1] + 2)
    ax[0].set_ylabel('#Patients', fontsize=18)
    ax[1].set_ylabel('#Patients', fontsize=18)
    plt.xlabel('Significant electrodes', fontsize=18)
    fig.savefig(f'F:/maryam_sh/integrated_grad/hist_oversampling_29p_separate.svg')
    fig.savefig(f'F:/maryam_sh/integrated_grad/hist_oversampling_29p_separate.png')


def plot_markers_heatmap(patient, trial_id, T):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')
    title_list = ['Music_trial', 'Singing_trial']
    n_row = 2

    fig, ax = plt.subplots(n_row, int(ig[patient][0].shape[0] / T), figsize=(40, 9 * n_row),
                           dpi=300)
    for i in range(n_row):
        for time in range(int(ig[patient][0, :, :].shape[0] / T)):
            data_in = preprocessing.normalize(
                np.mean(ig[patient][trial_id[i]:trial_id[i + 1], :, :], axis=0)).T[:len(node_coords[patient]), time * T]
            plotting.plot_markers(data_in, node_coords[patient], node_size='auto', node_cmap='RdPu',
                                  alpha=0.7, output_file=None, display_mode=ch[patient][0][4], axes=ax[i, time],
                                  annotate=True, black_bg=False, colorbar=True, radiological=False)

            ax[i, time].set_title(f'T={time * T}', fontsize=45, pad=0)

        fig.text(0.06, 0.72 - 0.45 * i, title_list[i], fontsize=46, ha='center', weight='bold')

    fig.savefig(f'F:/maryam_sh/integrated_grad/markers_heatmap_patient_{patient + 1}.png')
    fig.savefig(f'F:/maryam_sh/integrated_grad/markers_heatmap_patient_{patient + 1}.svg')


def find_best_lobe():
    event_id_f = 0
    num_best_elec = 1
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')

    pos_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f))
    label_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f), dtype='U30')
    coor_best_elec_right = np.zeros((11, 3))
    coor_best_elec_left = np.zeros((18, 3))
    m = 0
    n = 0
    best = []
    for patient in range(len(ig)):
        for event in range(event_id_f, len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event - event_id_f] = pos
            label_elec_best[patient, event - event_id_f] = ch[patient][pos]
        best_elec = np.array(Counter(label_elec_best[patient, :]).most_common())[0:num_best_elec, 0].tolist()
        pos_best_elec = np.where(np.array(ch[patient]) == best_elec)[0][0]
        best.append(best_elec)
        if ch[patient][0][4] == 'r':
            a = []
            for i in range(len(np.where(np.array(ch[patient]) == best_elec)[0])):
                a.append(node_coords[patient][np.where(np.array(ch[patient]) == best_elec)[0][i], :])
            np.save(f'F:/maryam_sh/integrated_grad/{patient}_coor_best.npy', a)
            coor_best_elec_right[m, :] = node_coords[patient][pos_best_elec, :]
            m = m + 1
        else:
            a = []
            for i in range(len(np.where(np.array(ch[patient]) == best_elec)[0])):
                a.append(node_coords[patient][np.where(np.array(ch[patient]) == best_elec)[0][i], :])
            np.save(f'F:/maryam_sh/integrated_grad/{patient}_coor_best.npy', a)
            coor_best_elec_left[n, :] = node_coords[patient][pos_best_elec, :]
            n = n + 1

    np.save('F:/maryam_sh/integrated_grad/best_lobes.npy', best)


def plot_heatmap_best_lobe():
    with open('/content/ch_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    best = np.load("/content/best.npy")

    num_fig_each_row = 6
    fig, ax = plt.subplots(5, 6, dpi=300)
    plt.subplots_adjust(hspace=0.5)
    plt.axis('off')
    fig_id = 0
    for patient in range(29):
        coor_best = np.load(f"/content/{patient}.npy")
        data_in = []
        for i in range(len(coor_best)):
            data_in.append(1)
        plotting.plot_markers(data_in, coor_best, node_size=30, node_cmap='Wistia',
                              alpha=1, output_file=None, display_mode=ch[patient][0][4],
                              axes=ax[int(fig_id / num_fig_each_row), fig_id % num_fig_each_row],
                              annotate=True, black_bg=False, colorbar=False, radiological=False)
        fig_id += 1
        plt.title(f'Patient {patient + 1} \n{best[patient][0][7:]}', fontsize=5)

    plt.savefig("plot_heatmap_best_lobe.png")
    plt.savefig("plot_heatmap_best_lobe.svg")

# plot_elec_point(patient=0)
# plot_heatmap_per_patient(patient_id=0, trial_id=[0, 73, 89])
# plot_histogram(event_id_f=0, num_best_elec=3)
# plot_markers_heatmap(patient=13, trial_id=[0, 73, 89], T=40)
# plot_hist_seperate(num_best_elec=1)
# find_best_lobe()
# plot_heatmap_best_lobe()
