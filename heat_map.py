import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pickle as pkl
from collections import Counter
import scipy.io
import mne

def plot_hist(type):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)
    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_{type}_29p.npy')

    pos_elec_best = np.zeros((len(ig), len(ig[0])))
    label_elec_best = np.zeros((len(ig), len(ig[0]),), dtype='U30')
    count_label = []
    for patient in range(len(ig)):
        for event in range(len(ig[0])):
            data_norm = preprocessing.normalize(ig[patient][event])
            pos = np.argmax(np.sum(data_norm, axis=0))
            pos_elec_best[patient, event] = pos
            label_elec_best[patient, event] = ch[patient][pos][7:]
        count_label.append(Counter(label_elec_best[patient, :]).most_common()[0][0])

    print(Counter(count_label))
    plt.figure(dpi=300)
    color_list = ['yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'pink', 'darkgray', 'orangered','forestgreen', 'navy', 'forestgreen']

    for i in range(len(Counter(count_label))):
        plt.bar(i, Counter(count_label).most_common()[i][1], color=color_list[i], ecolor='blue', capsize=3,
                     label=Counter(count_label).most_common()[i][0])

    plt.legend()
    plt.xticks([])
    plt.ylim(0,10)
    plt.ylabel('#Patients')
    plt.xlabel('Significant electrodes')
    plt.title(f'Histogram significant electrodes across 29 patients ({type} model)', fontsize=9)
    plt.savefig(f'F:/maryam_sh/integrated_grad/hist_{type}_29p.png')
    print('end')

def plot_heat_map(patient, event_list, T, type):
    with open('F:/maryam_sh/load_data/new_dataset/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)

    elec_coor = np.load('F:/maryam_sh/new_dataset/dataset/Singing_Music/elec_coor_patients.npy', allow_pickle=True)

    ig = np.load(f'F:/maryam_sh/integrated_grad/Ig_{type}_29p.npy')

    cmap_list=['RdPu', 'Purples']
    title_list=['Music_trial','Singing_trial']
    ch_names = []
    for i in range(len(ch[patient])):
        ch_names.append(str(i) + '_' + ch[patient][i][7:])

    if ch[patient][0][4]=='r':
        hemis='Right'
    else: hemis='Left'

    info = mne.create_info(ch_names, sfreq=1000, ch_types='eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec_coor[patient]/1100))))
    fig, ax = plt.subplots(len(event_list), int(ig[patient][0].shape[0]/T), figsize=(40, 9*len(event_list)), dpi=300)
    for i, event in enumerate(event_list):
        for time in range(int(ig[patient][0].shape[0] / T)):
            data_in=preprocessing.normalize(ig[patient][event]).T[:len(ch[patient]),time*T]
            # Plot electrode positions
            im, _ =mne.viz.plot_topomap(data_in, info, res=100, size=5, cmap=cmap_list[i],axes=ax[i,time],
                                        show=False, vlim=[-1,1])
            ax[i, time].set_title(f'T={time*T}', fontsize=45, pad=0)


        fig.text(0.06, 0.72-0.45*i, title_list[i], fontsize=46, ha='center', weight='bold')
        cbar_ax = fig.add_axes([0.92, 0.6-0.45*i, 0.03, 0.3])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.outline.set_edgecolor(cbar.cmap(0.5))
        cbar.outline.set_linewidth(3)
        cbar.set_label('Integrated Gradient', fontsize=30)
        cbar.ax.tick_params(labelsize=30)

    plt.suptitle(f'Temporal Variation of Spatial Features in patient_{patient+1} with {hemis} hemisphere electrodes',
                 fontsize=30,weight='bold')
    fig.savefig(f'F:/maryam_sh/integrated_grad/heat_map_patient_{patient+1}.png')

# plot_hist(type='oversampling')
plot_heat_map(patient=12, event_list=[0,73], T=40, type='oversampling')
