import os
import pickle as pkl
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# Set environment variable for thread management
os.environ["OMP_NUM_THREADS"] = "1"




def select_random_event(num_minority, num_majority, task, random_seed):
    """Select random events to balance classes."""
    random.seed(random_seed)
    if task == 'Question_Answer':
        inds_ran = random.sample(range(num_minority, num_majority + num_minority), num_minority)
        inds_ran.extend(range(0, num_minority))
    else:
        inds_ran = random.sample(range(0, num_majority), num_minority)
        inds_ran.extend(range(num_majority, num_majority + num_minority))
    return inds_ran


def folds_choose(settings, labels, num_events, num_minority, num_majority, random_seed=42):
    """Split events into train, validation, and test indices."""
    inds_all_train, inds_all_val, inds_all_test = [], [], []
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)

    if settings.type_balancing != 'under_sampling':
        events_order = np.arange(num_events)
    for fold in range(settings.n_folds):
        if settings.type_balancing == 'under_sampling':
            events_order = np.array(select_random_event(num_minority, num_majority, settings.task, random_seed))
        else:
            np.random.shuffle(events_order)

        if settings.one_patient_out:
            for train_index, val_index in stratified_splitter.split(np.zeros_like(labels[events_order]),
                                                                    labels[events_order]):
                inds_all_val.append(events_order[val_index])
                inds_all_train.append(events_order[train_index])
                inds_all_test.append(events_order[val_index])
        else:
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


def zeropad_data(x_train_all, x_test_all, x_val_all, num_input):
    """Zero-pad data"""
    x_train_zero_all = []
    x_test_zero_all = []
    x_val_zero_all = []
    for num in range(num_input):
        x_train_zero = np.zeros(
            ((x_train_all[0].shape[0]) * num_input, x_train_all[num].shape[1], x_train_all[num].shape[2]))
        x_train_zero[(x_train_all[0].shape[0]) * num:(x_train_all[0].shape[0]) * (num + 1), :, :] = x_train_all[num]
        x_train_zero_all.append(x_train_zero)

        x_test_zero = np.zeros(
            ((x_test_all[0].shape[0]) * num_input, x_test_all[num].shape[1], x_test_all[num].shape[2]))
        x_test_zero[(x_test_all[0].shape[0]) * num:(x_test_all[0].shape[0]) * (num + 1), :, :] = x_test_all[num]
        x_test_zero_all.append(x_test_zero)

        x_val_zero = np.zeros(((x_val_all[0].shape[0]) * num_input, x_val_all[num].shape[1], x_val_all[num].shape[2]))
        x_val_zero[(x_val_all[0].shape[0]) * num:(x_val_all[0].shape[0]) * (num + 1), :, :] = x_val_all[num]
        x_val_zero_all.append(x_val_zero)

    return x_train_zero_all, x_test_zero_all, x_val_zero_all



