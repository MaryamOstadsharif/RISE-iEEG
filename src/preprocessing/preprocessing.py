import xarray as xr
from .utils import del_temporal_lobe
from src.preprocessing.import_data_audio_visual import *
from collections import Counter

def time_ann(path):
    r = pd.read_csv(path, sep=";")
    onset = []
    offset = []
    for i in range(len(r.index)):
        d = r.iloc[i, 0]
        pos1 = d.find('\t')
        pos2 = d.rfind('\t')
        onset.append(eval(d[pos1 + 1:pos2]))
        offset.append(eval(d[pos2 + 1:]))
    return onset, offset

def read_time(paths):
    onset_1, offset_1 = time_ann(path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_questions.tsv")
    onset_all, offset_all = time_ann(path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_sentences.tsv")

    # remove onset of question from onset of answer
    onset_1_int = [int(x) for x in onset_1]
    offset_1_int = [int(x) for x in offset_1]

    for i in onset_all:
        if int(i) in onset_1_int:
            onset_all.remove(i)

    for i in onset_all:
        if i in onset_1:
            onset_all.remove(i)

    for i in offset_all:
        if int(i) in offset_1_int:
            offset_all.remove(i)

    for i in offset_all:
        if i in offset_1:
            offset_all.remove(i)

    onset_0 = onset_all
    offset_0 = offset_all

    return onset_1, onset_0

class DataPreprocessor:
    def __init__(self, settings, paths):
        self.settings = settings
        self.paths = paths

    def load_or_preprocess(self):
        """Load preprocessed data if available, otherwise preprocess and save."""
        # Path to preprocessed data
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)

        # Check if preprocessed data exists
        preprocessed_file_path = os.path.join(save_path, 'labels.pkl')
        if os.path.exists(preprocessed_file_path) and self.settings.load_preprocessed_data:
            print('Preprocessed data found. Loading data...')
            return self._load_preprocessed_data(save_path)
        else:
            print('Preprocessed data not found or not requested. Preprocessing data...')
            self.preprocess_and_save()
            return self._load_preprocessed_data(save_path)

    def preprocess_and_save(self):
        """Preprocess data based on the task and save it."""
        if self.settings.task == 'Speech_Music' or self.settings.task == 'Question_Answer':
            self._process_audio_visual()
        elif self.settings.task == 'Singing_Music':
            self._process_music_reconstruction()
        elif self.settings.task == 'Move_Rest':
            self._process_upper_limb_movement()
        else:
            raise ValueError(f"Unsupported dataset: {self.settings.task}")

    def _load_preprocessed_data(self, save_path):
        """Load preprocessed data from files."""
        data_all_input = []
        for patient in range(self.settings.num_patient):
            file_path = os.path.join(save_path, f'patient_{patient + 1}_reformat.pkl')
            with open(file_path, 'rb') as f:
                data_all_input.append(pkl.load(f))

        label_file_path = os.path.join(save_path, 'labels.pkl')
        with open(label_file_path, 'rb') as f:
            labels = pkl.load(f)

        if self.settings.del_temporal_lobe is True:
            # delete Superior temporal lobe data
            print("====== Experimental settings: Superior temporal lobe data is deleted ======")
            data_all_input = del_temporal_lobe(path=save_path,
                                               data=data_all_input,
                                               task=self.settings.task)

        return data_all_input, labels

    def _process_audio_visual(self):
        """Preprocess the audio-visual data."""
        dataset = 'Audio_visual'
        root_path = os.path.join(self.paths.raw_dataset_path, dataset)
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        os.makedirs(save_path, exist_ok=True)

        band_all_patient_with_hilbert, _, _ = get_data(root_path, settings=self.settings)

        if self.settings.task == 'Speech_Music':
            time_1 = np.arange(0, 30, 2)
            time_0 = np.arange(30, 60, 2)
            onset_1, onset_0 = [], []
            for i in range(7):
                onset_1.extend((time_1 + i * 60).tolist())
                if i < 6:
                    onset_0.extend((time_0 + i * 60).tolist())
            t_min, t_max = 0, 2
        elif self.settings.task == 'Question_Answer':
            onset_1, onset_0 = read_time(paths=self.paths.raw_dataset_path)
            t_min, t_max = 0.5, 2.5
        else:
            raise ValueError(f"Unsupported task: {self.settings.task}")

        onset = onset_1 + onset_0
        fs = 25

        for patient in range(self.settings.num_patient):
            file_path = os.path.join(save_path, f'patient_{patient + 1}_reformat.pkl')
            print(f'Reformatting data for patient {patient + 1}')
            data = band_all_patient_with_hilbert[patient]['gamma'].T
            data_reformat = np.zeros((len(onset_1) + len(onset_0), data.shape[0], int((t_min + t_max) * fs)))
            for event in range(len(onset)):
                start_sample = int((onset[event] - t_min) * fs)
                end_sample = int((onset[event] + t_max) * fs)
                data_reformat[event, :, :] = data[:, start_sample:end_sample]

            with open(file_path, 'wb') as f:
                pkl.dump(data_reformat, f)

        labels = [1] * len(onset_1) + [0] * len(onset_0)
        label_save_path = os.path.join(save_path, 'labels.pkl')
        if not os.path.exists(label_save_path):
            with open(label_save_path, 'wb') as f:
                pkl.dump(np.array(labels), f)

        print('Audio-Visual data preprocessing complete.')

    def _process_music_reconstruction(self):
        """Preprocess the music reconstruction data."""
        dataset = 'Music_Reconstruction/data_all_patient.pkl'
        root_path = os.path.join(self.paths.raw_dataset_path, dataset)
        with open(root_path, 'rb') as f:
            data_all_patient = pkl.load(f)

        onset_0 = [14, 16, 24, 26, 28, 34, 36, 43, 45, 47, 56, 57, 64, 66, 73, 75]
        onset_1 = [i for i in np.arange(0, 14, 2)] + [19, 21, 30, 32] + [40] + [i for i in np.arange(50, 56, 2)] + \
                  [60, 62, 69, 71] + [i for i in np.arange(81, 189, 2)]
        onset_1[0] += 0.5
        onset = onset_1 + onset_0
        t_min, t_max = 0.5, 1.5
        fs = 100
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        os.makedirs(save_path, exist_ok=True)

        for patient in range(len(data_all_patient)):
            file_path = os.path.join(save_path, f'patient_{patient + 1}_reformat.pkl')
            print(f'Reformatting data for patient {patient + 1}')
            data = data_all_patient[patient]['gamma'].T
            data_reformat = np.zeros((len(onset_1) + len(onset_0), data.shape[0], int((t_min + t_max) * fs)))
            for event in range(len(onset)):
                start_sample = int((onset[event] - t_min) * fs)
                end_sample = int((onset[event] + t_max) * fs)
                data_reformat[event, :, :] = data[:, start_sample:end_sample]

            with open(file_path, 'wb') as f:
                pkl.dump(data_reformat, f)
        labels = len(onset_1)*[1] + len(onset_0)*[0]
        file_path = os.path.join(save_path, 'labels.pkl')
        with open(file_path, 'wb') as f:
            pkl.dump(np.array(labels), f)

        print('Music reconstruction data preprocessing complete.')

    def _process_upper_limb_movement(self):
        """Preprocess the upper limb movement data."""
        dataset = 'Upper_Limb_Movement'
        rootpath = os.path.join(self.paths.raw_dataset_path, dataset)
        pats_ids_in = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12']
        tlim = [-1, 1]
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        os.makedirs(save_path, exist_ok=True)

        for j in range(len(pats_ids_in)):
            file_path = os.path.join(save_path, f'patient_{j + 1}_reformat.pkl')
            print(f'Reformatting data for patient {j + 1}')
            pat_curr = pats_ids_in[j]
            ep_data_in = xr.open_dataset(os.path.join(rootpath, pat_curr + '_ecog_data.nc'))
            ep_times = np.asarray(ep_data_in.time)
            time_inds = np.nonzero(np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1]))[0]
            n_ecog_chans = (len(ep_data_in.channels) - 1)

            days_all_in = np.asarray(ep_data_in.events)
            days_train_inds = []
            for day_tmp in list(np.unique(days_all_in)):
                days_train_inds.extend(np.nonzero(days_all_in == day_tmp)[0])

            dat = ep_data_in[dict(events=days_train_inds, channels=slice(0, n_ecog_chans), time=time_inds)].to_array().values.squeeze()
            labels = ep_data_in[dict(events=days_train_inds, channels=ep_data_in.channels[-1], time=0)].to_array().values.squeeze() - 1

            data_reformat = np.zeros((300, dat.shape[1], dat.shape[2]))
            n = 0
            for k in range(dat.shape[0]):
                if labels[k] == 0 and n < 150:
                    data_reformat[n, :, :] = dat[k, :, :]
                    n += 1

            for k in range(dat.shape[0]):
                if labels[k] == 1 and n < 300:
                    data_reformat[n, :, :] = dat[k, :, :]
                    n += 1

            with open(file_path, 'wb') as f:
                pkl.dump(data_reformat, f)

        label_save_path = os.path.join(save_path, 'labels.pkl')
        if not os.path.exists(label_save_path):
            labels = np.array([0] * 150 + [1] * 150)
            with open(label_save_path, 'wb') as f:
                pkl.dump(labels, f)

        print('Move-rest data preprocessing complete.')
