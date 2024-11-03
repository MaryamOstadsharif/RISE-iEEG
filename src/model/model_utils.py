import os
import pickle as pkl
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# Set environment variable for thread management
os.environ["OMP_NUM_THREADS"] = "1"


def folds_choose(settings, labels, stage, num_folds, random_seed):
    """Split events into train, validation, and test indices."""
    inds_all_train, inds_all_val, inds_all_test = [], [], []
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)

    events_order = np.arange(len(labels))
    for fold in range(num_folds):
        np.random.shuffle(events_order)
        if settings.mode == 'Unseen_patient' and stage == 'First_train':
            for train_index, val_index in stratified_splitter.split(np.zeros_like(labels[events_order]),
                                                                    labels[events_order]):
                inds_all_val.append(events_order[val_index])
                inds_all_train.append(events_order[train_index])
        if settings.mode == 'Same_patient' or stage == 'Second_train':
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
    num_majority_class = Counter(y_train).most_common()[0][1]
    num_minority_class = Counter(y_train).most_common()[1][1]
    inds_tmp_orig = np.arange(0, len(y_train))
    inds_tmp = inds_tmp_orig[y_train == Counter(y_train).most_common()[1][0]]
    inds_tmp = list(inds_tmp) * ((num_majority_class // len(inds_tmp)) + 1)
    y_train = np.concatenate((y_train, y_train[inds_tmp[:num_majority_class - num_minority_class]]), axis=0)
    for i in range(len(x_train_all)):
        print("\n Input_", str(i))
        x_train_all[i] = np.concatenate(
            (x_train_all[i], x_train_all[i][inds_tmp[:num_majority_class - num_minority_class], :, :]), axis=0)

    return x_train_all, y_train


def zeropad_data(data):
    """Zero-pad data"""
    num_input = len(data)
    data_pad = []
    for num in range(num_input):
        data_zero = np.zeros(((data[0].shape[0]) * num_input, data[num].shape[1], data[num].shape[2]))
        data_zero[(data[0].shape[0]) * num:(data[0].shape[0]) * (num + 1), :, :] = data[num]
        data_pad.append(data_zero)

    return data_pad
