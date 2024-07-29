import pickle as pkl
import numpy as np
from import_data import *
from utils import read_time

# task : 'speech&music' , 'question&answer' for dataset: 'audio_visual'
task = 'question&answer'
# dataset: 'audio_visual', 'music_reconstruction'
dataset = 'audio_visual'

if dataset=='audio_visual':
    class Paths:
        def __init__(self):
            self.path_processed_data = ''
            self.path_dataset = ''

        def create_path(self, path_dataset, path_processed_data):
            self.path_dataset = path_dataset
            self.path_processed_data = path_processed_data


    # set device
    device = 'system_lab'
    if device.lower() == 'navid':
        dataset_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
        processed_data_path = 'F:/maryam_sh/load_data/'
    elif device.lower() == 'maryam':
        dataset_path = 'E:/Thesis/dataset/dataset/'
        processed_data_path = 'E:/Thesis/derived data/'
    elif device.lower() == 'system_lab':
        dataset_path = 'F:/maryam_sh/dataset/'
        processed_data_path = 'F:/maryam_sh/load_data/'
    elif device.lower() == 'navid_lab':
        dataset_path = 'D:/Navid/Dataset/AudioVisualiEEG/'
        processed_data_path = 'D:/Navid/Dataset/AudioVisualiEEG/processed_data/'
    else:
        dataset_path = ''
        processed_data_path = ''

    load_data_settings = {
        'dataset': 'audio_visual',  # dataset: 'audio_visual', 'music_reconstruction'
        'number_of_patients': 63,  # if dataset is 'audio_visual':'number_of_patients'= 63,
        # if dataset is'music_reconstruction':'number_of_patients'= 29
        # if 'load_preprocessed_data':False, function create preprocessed_data, else it just load data
        'load_preprocessed_data': True,
        'save_preprocessed_data': True
    }
    paths = Paths()
    paths.create_path(path_dataset=dataset_path,
                      path_processed_data=processed_data_path)

    band_all_patient_with_hilbert, band_all_patient_without_hilbert, channel_names_list = \
        get_data(paths, settings=load_data_settings)

    if task=='speech&music':
        time_1 = np.arange(0, 30, 2)
        time_0 = np.arange(30, 60, 2)
        onset_1 = []
        onset_0 = []
        for i in range(7):
            onset_1.extend((time_1 + i * 60).tolist())
            if i < 6:
                onset_0.extend((time_0 + i * 60).tolist())

        t_min = 0
        t_max = 2

    if task=='question&answer':
        onset_1, onset_0 = read_time(task=task,
                                     t_min=0.5,
                                     paths=paths)
        t_min = 0.5
        t_max = 2.5

    onset = onset_1 + onset_0


    fs = 25
    for patient in range(len(band_all_patient_with_hilbert)):
        print('reformat data patient_', str(patient))
        data = band_all_patient_with_hilbert[patient]['gamma'].T
        data_reformat = np.zeros((len(onset_1) + len(onset_0), data.shape[0], int((t_min + t_max) * fs)))
        for event in range(len(onset)):
            start_sample = int(onset[event] - t_min) * fs
            end_sample = int(onset[event] + t_max) * fs
            data_reformat[event, :, :] = data[:, start_sample:end_sample]

        with open('F:/maryam_sh/new_dataset/dataset_audiovisual_reformat/'+task+'/patient_' + str(patient + 1) + '_reformat.pkl', 'wb') as f:
            pkl.dump(data_reformat, f)

    label= [1]*len(onset_1)+[0]*len(onset_0)
    with open('F:/maryam_sh/new_dataset/dataset_audiovisual_reformat/' + task + '/label.pkl', 'wb') as f:
        pkl.dump(label, f)

    print('end')
