import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle as pkl
from collections import Counter
from nilearn import plotting


def plot_heatmap_matrix(patient_id, event):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_{type}_29p.npy')
    data_norm = preprocessing.normalize(ig[patient_id][event])
    plt.figure(dpi=300)
    plt.imshow(data_norm[:, :len(ch[patient_id])], cmap='viridis')
    plt.colorbar()
    plt.title(f'Integrated Gradient patient_{patient_id}/ event_{event}')
    plt.savefig(f'F:/maryam_sh/integrated_grad/ht_{patient_id}.png')


def plot_hist(type,event_id_f):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_{type}_29p.npy')


    pos_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f))
    label_elec_best = np.zeros((len(ig), len(ig[0]) - event_id_f), dtype='U30')
    count_label = []
    for patient in range(len(ig)):
        for event in range(event_id_f, len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event - event_id_f] = pos
            label_elec_best[patient, event - event_id_f] = ch[patient][pos][7:]
        count_label.append(Counter(label_elec_best[patient, :]).most_common()[0][0])

    print(Counter(count_label))
    plt.figure(dpi=300)
    color_list = ['yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'pink', 'darkgray', 'orangered',
                  'forestgreen', 'navy', 'forestgreen']

    for i in range(len(Counter(count_label))):
        plt.bar(i, Counter(count_label).most_common()[i][1], color=color_list[i], ecolor='blue', capsize=3,
                label=Counter(count_label).most_common()[i][0])

    plt.legend()
    plt.xticks([])
    plt.ylim(0,Counter(count_label).most_common()[0][1]+2)
    plt.ylabel('#Patients')
    plt.xlabel('Significant electrodes')
    plt.title(f'Histogram significant electrodes across 29 patients ({type} model) for {len(ig[0]) - event_id_f} events', fontsize=9)
    plt.savefig(f'F:/maryam_sh/integrated_grad/hist_{type}_29p.png')
    print('end')


def plot_heat_map(patient, number_trial, T, type):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    node_coords = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_{type}_29p.npy')
    title_list = ['Music_trial', 'Singing_trial']
    n_row=2

    if ch[patient][0][4] == 'r':
        hemis = 'Right'
    else:
        hemis = 'Left'

    fig, ax = plt.subplots(n_row, int(ig[patient][0].shape[0] / T), figsize=(40, 9 * n_row),
                           dpi=300)
    for i in range(n_row):
        for time in range(int(ig[patient][0, :, :].shape[0] / T)):
            data_in = preprocessing.normalize(
                np.mean(ig[patient][number_trial[i]:number_trial[i + 1], :, :], axis=0)).T[:len(node_coords[patient]),
                      time * T]
            plotting.plot_markers(data_in, node_coords[patient], node_size='auto', node_cmap='RdPu',
                                  alpha=0.7, output_file=None, display_mode=ch[patient][0][4], axes=ax[i, time],
                                  annotate=True,black_bg=False, colorbar=True, radiological=False)

            ax[i, time].set_title(f'T={time * T}', fontsize=45, pad=0)

        fig.text(0.06, 0.72 - 0.45 * i, title_list[i], fontsize=46, ha='center', weight='bold')

    plt.suptitle(f'Temporal Variation of Spatial Features in patient_{patient + 1} with {hemis} hemisphere electrodes',
                 fontsize=30, weight='bold')
    fig.savefig(f'F:/maryam_sh/integrated_grad/heat_map_patient_{patient + 1}.png')




# plot_heatmap_matrix(patient_id=2, event=5)
# plot_hist(type='oversampling', event_id_f=0)
plot_heat_map(patient=0, number_trial=[0,73,89], T=40, type='oversampling')

