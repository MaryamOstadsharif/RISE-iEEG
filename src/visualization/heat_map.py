import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle as pkl
from collections import Counter
from nilearn import plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_elec_point(patient):
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    coor_elec = node_coords[patient]
    plot = plotting.view_markers(coor_elec, marker_size=10)
    plot.save_as_html(f'F:/maryam_sh/integrated_grad/plot_elec_point_patient{patient}.html')


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

def plot_histogram(event_id_f, num_best_elec, del_lobe):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)

    if del_lobe:
        for patient in range(29):
            del_ind = []
            for pos, channel in enumerate(ch[patient]):
                if channel[7:] == 'superiortemporal':
                    del_ind.append(pos)
            ch[patient] = np.delete(ch[patient], del_ind)
        with open('F:/maryam_sh/integrated_grad/Ig_oversampling_del_temporal.pkl', 'rb') as f:
            ig = pkl.load(f)

    else:
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
    color_list = ['steelblue', 'deeppink', 'cyan', 'purple', 'lime', 'yellow', 'red', 'darkgray', 'brown',
                  'forestgreen', 'navy', 'forestgreen', 'blue', 'violet', 'red', 'brown', 'green', 'gray', 'darkblue']

    if del_lobe:
        list_lobe = ['superiortemporal', 'rostralmiddlefrontal', 'middletemporal', 'postcentral',
                     'precentral', 'caudalmiddlefrontal', 'parsopercularis', 'inferiortemporal',
                     'supramarginal', 'superiorfrontal']
        count = Counter(count_label)
        count.update({'superiortemporal': 0})
        for i, lobe in enumerate(list_lobe):
            plt.bar(i, (Counter(count_label)[lobe]/29)*100, color=color_list[i], ecolor='blue', capsize=3, label=lobe)

    else:
        for i in range(10):
            plt.bar(i, (Counter(count_label).most_common()[i][1]/29)*100, color=color_list[i], ecolor='blue', capsize=3,
                    label=Counter(count_label).most_common()[i][0])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.legend()
    plt.xticks(range(10),['ST', 'RMF', 'MT', 'PC', 'PrC', 'CMF', 'PO', 'IT', 'SM', 'SF'], fontsize=12)
    plt.ylim(0, 90)
    plt.yticks(np.arange(0,100,10), fontsize=12)
    plt.ylabel('Percentage of patients', fontsize=12)
    plt.xlabel('Significant lobes', fontsize=12)
    # plt.title(f'Histogram significant electrodes across 29 patients for {len(ig[0]) - event_id_f} events', fontsize=9)
    plt.tight_layout()
    if del_lobe:
        plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p_del.png')
        plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p_del.svg')
    else:
        plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p.png')
        plt.savefig(f'F:/maryam_sh/integrated_grad/hist_29p.svg')


def plot_markers_heatmap(patient, trial_id, T):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_oversampling_29p.npy')
    title_list = ['Music event', 'Singing event']
    n_col = 2

    fig, ax = plt.subplots(int(ig[patient][0, :, :].shape[0] / T) + 1, n_col, figsize=(20, 25), dpi=300)

    for i in range(n_col):
        for time in range(int(ig[patient][0, :, :].shape[0] / T) + 1):
            data_in = preprocessing.normalize(
                np.mean(ig[patient][trial_id[i]:trial_id[i + 1], :, :], axis=0)).T[:len(node_coords[patient]),
                      time * T - 1]
            plot = plotting.plot_markers(data_in, node_coords[patient], node_size=120, node_cmap='bwr',
                                         alpha=1, output_file=None, display_mode=ch[patient][0][4], axes=ax[time, i],
                                         annotate=True, black_bg=False, colorbar=True, radiological=False, node_vmin=0)

            # Adjust the colorbar font size
            cbar = plot._cbar
            cbar.ax.tick_params(labelsize=20)

            fig.text(0.01, 0.82 - 0.13 * time, f'T= {time * T / 100:g} sec', fontsize=30, ha='left', weight='bold')

        ax[0, i].set_title(title_list[i], fontsize=60, pad=100, weight='bold')

    # fig.tight_layout()
    fig.savefig(f'F:/maryam_sh/integrated_grad/markers_heatmap_patient_{patient}.png')
    fig.savefig(f'F:/maryam_sh/integrated_grad/markers_heatmap_patient_{patient}.svg')


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
            np.save(f'F:/maryam_sh/integrated_grad/best/3/{patient}_coor_best.npy', a)
            coor_best_elec_right[m, :] = node_coords[patient][pos_best_elec, :]
            m = m + 1
        else:
            a = []
            for i in range(len(np.where(np.array(ch[patient]) == best_elec)[0])):
                a.append(node_coords[patient][np.where(np.array(ch[patient]) == best_elec)[0][i], :])
            np.save(f'F:/maryam_sh/integrated_grad/best/3/{patient}_coor_best.npy', a)
            coor_best_elec_left[n, :] = node_coords[patient][pos_best_elec, :]
            n = n + 1

    np.save('F:/maryam_sh/integrated_grad/best/3/best_lobes.npy', best)


def plot_heatmap_best_lobe():
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    best = np.load("F:/maryam_sh/integrated_grad/old3/best.npy")

    num_fig_each_row = 6
    fig, ax = plt.subplots(5, 6, dpi=300)
    plt.subplots_adjust(hspace=0.5)
    plt.axis('off')
    fig_id = 0
    for patient in range(29):
        coor_best = np.load(f"F:/maryam_sh/integrated_grad/old3/{patient}.npy")
        data_in = []
        for i in range(len(coor_best)):
            data_in.append(1)
        plotting.plot_markers(data_in, coor_best, node_size=30, node_cmap='Wistia',
                              alpha=1, output_file=None, display_mode=ch[patient][0][4],
                              axes=ax[int(fig_id / num_fig_each_row), fig_id % num_fig_each_row],
                              annotate=True, black_bg=False, colorbar=False, radiological=False)

        fig_id += 1
        plt.title(f'Patient {patient + 1} \n{best[patient][0][7:]}', fontsize=5)
    plt.tight_layout()
    plt.savefig("F:/maryam_sh/integrated_grad/plot_heatmap_best_lobe.png")
    plt.savefig("F:/maryam_sh/integrated_grad/plot_heatmap_best_lobe.svg")


def plot_heatmap_best_lobe_per_patient():
    num_patient = 29
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    best = np.load("F:/maryam_sh/integrated_grad/old3/best.npy")

    for patient in range(num_patient):
        coor_best = np.load(f"F:/maryam_sh/integrated_grad/old3/{patient}.npy")
        data_in = []
        for i in range(len(coor_best)):
            data_in.append(1)

        plt.figure(figsize=(5, 5), dpi=300)
        plotting.plot_markers(data_in, coor_best, node_size=400, node_cmap='winter',
                              alpha=1, output_file=None, display_mode=ch[patient][0][4],
                              annotate=True, black_bg=False, colorbar=False, radiological=False)

        plt.title(f'Patient {patient + 1}({best[patient][0][7:].capitalize()} lobe)', fontsize=10)
        # plt.tight_layout()
        # plt.axis('off')
        plt.savefig(f"F:/maryam_sh/integrated_grad/plot_heatmap_best_lobe_p{patient}.png", bbox_inches='tight')
        plt.savefig(f"F:/maryam_sh/integrated_grad/plot_heatmap_best_lobe_p{patient}.svg", bbox_inches='tight')
        plt.close()

# plot_elec_point(patient=0)
# plot_heatmap_per_patient(patient_id=0, trial_id=[0, 73, 89])
# plot_histogram(event_id_f=0, num_best_elec=3, del_lobe=False)
# plot_markers_heatmap(patient=0, trial_id=[0, 73, 89], T=40)
# find_best_lobe()
# plot_heatmap_best_lobe()
plot_heatmap_best_lobe_per_patient()