import os
import pickle as pkl
import numpy as np
import random
from collections import Counter
from mne import set_log_level
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit

os.environ["OMP_NUM_THREADS"] = "1"
set_log_level(verbose='ERROR')


def load_data(num_patient, lp, n_chans_all, task):
    data_all_input = []

    with open(lp +task+ '/labels.pkl', 'rb') as f:
        label = pkl.load(f)

    for patient in range(num_patient):
        print('patient_', str(patient))
        with open(lp + task+'/patient_'+str(patient+1) + '_reformat.pkl', 'rb') as f:
            data_one_patient = pkl.load(f)
        n_ecog_chans = data_one_patient.shape[1]

        # Pad data in electrode dimension if necessary
        if (num_patient > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(data_one_patient.shape)
            dat_sh[1] = n_chans_all
            # Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:, :n_ecog_chans, ...] = data_one_patient
            dat = X_pad.copy()

        data_all_input.append(dat)

    print('Data loaded!')

    return data_all_input, label


def select_random_event(num_minority, num_majority, task):
    if task == 'Question_Answer':
        inds_ran = random.sample(range(num_minority, num_majority + num_minority), num_minority)
        inds_ran.extend(range(0, num_minority))
    else:
        inds_ran = random.sample(range(0, num_majority), num_minority)
        inds_ran.extend(range(num_majority, num_majority + num_minority))
    return inds_ran


def folds_choose(n_folds, labels, num_events, num_minority, num_majority, type, task, random_seed):
    inds_all_train, inds_all_val, inds_all_test = [], [], []
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    if type == 'over_sampling':
        events_order = np.arange(num_events)
    for fold in range(n_folds):

        if type == 'under_sampling':
            events_order = np.array(select_random_event(num_minority, num_majority, task))
        if type == 'over_sampling':
            np.random.shuffle(events_order)
        for train_val_index, test_index in stratified_splitter.split(np.zeros_like(labels[events_order]),
                                                                     labels[events_order]):
            inds_all_test.append(events_order[test_index])
            for train_index, val_index in stratified_splitter.split(
                    np.zeros_like(labels[events_order[train_val_index]]),
                    labels[events_order[train_val_index]]):
                inds_all_train.append(events_order[train_val_index[train_index]])
                inds_all_val.append(events_order[train_val_index[val_index]])

    return inds_all_train, inds_all_val, inds_all_test


def balance(x_train_all, y_train):
    oversample = SMOTE()
    num_majority_class = Counter(y_train).most_common()[0][1]

    x_all_resample = []

    for i in range(len(x_train_all)):
        print("Balancing data fot data input_", str(i))
        x_resample = np.zeros((num_majority_class * 2, x_train_all[i].shape[1], x_train_all[i].shape[2]))
        for ch in range(x_train_all[i].shape[1]):
            x_resample[:, ch, :], y_train_res = oversample.fit_resample(x_train_all[i][:, ch, :], y_train)
        x_all_resample.append(x_resample)
    return x_all_resample, y_train_res


def zeropad_data(x_train_all, x_test_all, x_val_all, num_patient):
    x_train_zero_all = []
    x_test_zero_all = []
    x_val_zero_all = []
    for num in range(num_patient):
        x_train_zero = np.zeros(
            ((x_train_all[0].shape[0]) * num_patient, x_train_all[0].shape[1], x_train_all[0].shape[2]))
        x_train_zero[(x_train_all[0].shape[0]) * num:(x_train_all[0].shape[0]) * (num + 1), :, :] = x_train_all[num]
        x_train_zero_all.append(x_train_zero)

        x_test_zero = np.zeros(((x_test_all[0].shape[0]) * num_patient, x_test_all[0].shape[1], x_test_all[0].shape[2]))
        x_test_zero[(x_test_all[0].shape[0]) * num:(x_test_all[0].shape[0]) * (num + 1), :, :] = x_test_all[num]
        x_test_zero_all.append(x_test_zero)

        x_val_zero = np.zeros(((x_val_all[0].shape[0]) * num_patient, x_val_all[0].shape[1], x_val_all[0].shape[2]))
        x_val_zero[(x_val_all[0].shape[0]) * num:(x_val_all[0].shape[0]) * (num + 1), :, :] = x_val_all[num]
        x_val_zero_all.append(x_val_zero)

    return x_train_zero_all, x_test_zero_all, x_val_zero_all
