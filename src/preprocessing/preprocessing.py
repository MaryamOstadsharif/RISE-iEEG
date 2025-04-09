import os
import pickle as pkl
import numpy as np
import xarray as xr
from collections import Counter
from .utils import del_temporal_lobe


class DataPreprocessor:
    def __init__(self, settings, paths):
        self.settings = settings
        self.paths = paths

    def load_or_preprocess(self):
        """
        Load preprocessed data if available, otherwise preprocess and save it.
        """
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        preprocessed_file_path = os.path.join(save_path, 'labels.pkl')

        if os.path.exists(preprocessed_file_path) and self.settings.load_preprocessed_data:
            print('Preprocessed data found. Loading data...')
            return self._load_preprocessed_data(save_path)
        else:
            print('Preprocessed data not found or not requested. Preprocessing data...')
            self.preprocess_and_save()
            return self._load_preprocessed_data(save_path)

    def preprocess_and_save(self):
        """
        Preprocess data based on the task and save it.
        """
        if self.settings.task == 'Singing_Music':
            self._process_music_reconstruction()
        elif self.settings.task == 'Move_Rest':
            self._process_upper_limb_movement()
        else:
            raise ValueError(f"Unsupported dataset task: {self.settings.task}")

    def _load_preprocessed_data(self, save_path):
        """
        Load preprocessed data from the saved files.
        """
        data_all_input = []

        for patient in range(self.settings.num_patient):
            file_path = os.path.join(save_path, f'patient_{patient + 1}_reformat.pkl')
            with open(file_path, 'rb') as f:
                data_all_input.append(pkl.load(f))

        label_file_path = os.path.join(save_path, 'labels.pkl')
        with open(label_file_path, 'rb') as f:
            labels = pkl.load(f)

        if self.settings.del_temporal_lobe:
            print("====== Experimental settings: Superior temporal lobe data is deleted ======")
            data_all_input = del_temporal_lobe(
                path=save_path,
                data=data_all_input,
                task=self.settings.task
            )

        return data_all_input, labels

    def _process_music_reconstruction(self):
        """
        Preprocess data for the Singing_Music task.
        """
        dataset_file = 'Music_Reconstruction/data_all_patient.pkl'
        root_path = os.path.join(self.paths.raw_dataset_path, dataset_file)
        with open(root_path, 'rb') as f:
            data_all_patient = pkl.load(f)

        # Define onsets
        onset_0 = [14, 16, 24, 26, 28, 34, 36, 43, 45, 47, 56, 57, 64, 66, 73, 75]
        onset_1 = (
            list(np.arange(0, 14, 2)) + [19, 21, 30, 32, 40] +
            list(np.arange(50, 56, 2)) + [60, 62, 69, 71] +
            list(np.arange(81, 189, 2))
        )
        onset_1[0] += 0.5
        onset = onset_1 + onset_0

        fs = 100
        t_min, t_max = 0.5, 1.5
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        os.makedirs(save_path, exist_ok=True)

        for patient, data_dict in enumerate(data_all_patient):
            file_path = os.path.join(save_path, f'patient_{patient + 1}_reformat.pkl')
            print(f'Reformatting data for patient {patient + 1}')

            data = data_dict['gamma'].T
            num_events = len(onset)
            data_reformat = np.zeros((num_events, data.shape[0], int((t_min + t_max) * fs)))

            for i, event_time in enumerate(onset):
                start_sample = int((event_time - t_min) * fs)
                end_sample = int((event_time + t_max) * fs)
                data_reformat[i, :, :] = data[:, start_sample:end_sample]

            with open(file_path, 'wb') as f:
                pkl.dump(data_reformat, f)

        # Save labels
        labels = [1] * len(onset_1) + [0] * len(onset_0)
        label_path = os.path.join(save_path, 'labels.pkl')
        with open(label_path, 'wb') as f:
            pkl.dump(np.array(labels), f)

        print('Music reconstruction data preprocessing complete.')

    def _process_upper_limb_movement(self):
        """
        Preprocess data for the Move_Rest task.
        """
        dataset_dir = 'Upper_Limb_Movement'
        root_path = os.path.join(self.paths.raw_dataset_path, dataset_dir)
        patient_ids = [f'EC{str(i).zfill(2)}' for i in range(1, 13)]
        tlim = [-1, 1]
        save_path = os.path.join(self.paths.preprocessed_dataset_path, self.settings.task)
        os.makedirs(save_path, exist_ok=True)

        for i, patient_id in enumerate(patient_ids):
            file_path = os.path.join(save_path, f'patient_{i + 1}_reformat.pkl')
            print(f'Reformatting data for patient {i + 1}')

            data_file = os.path.join(root_path, f'{patient_id}_ecog_data.nc')
            ep_data = xr.open_dataset(data_file)

            time_array = np.asarray(ep_data.time)
            time_inds = np.where((time_array >= tlim[0]) & (time_array <= tlim[1]))[0]
            n_channels = len(ep_data.channels) - 1

            event_ids = np.asarray(ep_data.events)
            train_inds = np.concatenate([
                np.where(event_ids == day)[0] for day in np.unique(event_ids)
            ])

            dat = ep_data[
                dict(events=train_inds, channels=slice(0, n_channels), time=time_inds)
            ].to_array().values.squeeze()

            labels_raw = ep_data[
                dict(events=train_inds, channels=ep_data.channels[-1], time=0)
            ].to_array().values.squeeze()
            labels = labels_raw - 1  # convert to 0/1

            # Balance and reformat data (150 samples per class)
            data_reformat = np.zeros((300, dat.shape[1], dat.shape[2]))
            n = 0
            for k in range(dat.shape[0]):
                if labels[k] == 0 and n < 150:
                    data_reformat[n] = dat[k]
                    n += 1
            for k in range(dat.shape[0]):
                if labels[k] == 1 and n < 300:
                    data_reformat[n] = dat[k]
                    n += 1

            with open(file_path, 'wb') as f:
                pkl.dump(data_reformat, f)

        # Save labels (if not already saved)
        label_file = os.path.join(save_path, 'labels.pkl')
        if not os.path.exists(label_file):
            labels = np.array([0] * 150 + [1] * 150)
            with open(label_file, 'wb') as f:
                pkl.dump(labels, f)

        print('Move-rest data preprocessing complete.')
