import os
import numpy as np
import pickle as pkl
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# Limit CPU threading for reproducibility and performance control
os.environ["OMP_NUM_THREADS"] = "1"


def folds_choose(settings, labels, stage, num_folds=1, random_seed=42):
    """
    Generate train, validation, and test indices using stratified splits.

    Args:
        settings: An object with a `.mode` attribute ('Unseen_patient' or 'Same_patient').
        labels: Array of class labels for stratification.
        stage: Either 'First_train' or 'Second_train'.
        num_folds: Number of folds to generate (typically 1).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of lists: (train_indices, val_indices, test_indices)
    """
    inds_all_train, inds_all_val, inds_all_test = [], [], []
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)

    events_order = np.arange(len(labels))

    for _ in range(num_folds):
        np.random.shuffle(events_order)

        if settings.mode == 'Unseen_patient' and stage == 'First_train':
            for train_idx, val_idx in strat_split.split(np.zeros_like(labels[events_order]), labels[events_order]):
                inds_all_train.append(events_order[train_idx])
                inds_all_val.append(events_order[val_idx])

        elif settings.mode == 'Same_patient' or stage == 'Second_train':
            for train_val_idx, test_idx in strat_split.split(np.zeros_like(labels[events_order]), labels[events_order]):
                inds_all_test.append(events_order[test_idx])
                for train_idx, val_idx in strat_split.split(np.zeros_like(labels[events_order[train_val_idx]]),
                                                            labels[events_order[train_val_idx]]):
                    inds_all_train.append(events_order[train_val_idx[train_idx]])
                    inds_all_val.append(events_order[train_val_idx[val_idx]])

    return inds_all_train, inds_all_val, inds_all_test


def balance(x_train_all, y_train):
    """
    Balance the dataset by oversampling the minority class.

    Args:
        x_train_all: List of input arrays for each modality or patient.
        y_train: 1D array of class labels.

    Returns:
        Tuple: (balanced x_train_all, balanced y_train)
    """
    class_counts = Counter(y_train)
    majority_class, majority_count = class_counts.most_common(1)[0]
    minority_class, minority_count = class_counts.most_common()[1]

    minority_indices = np.where(y_train == minority_class)[0]
    oversample_indices = np.tile(minority_indices,
                                 (majority_count // len(minority_indices) + 1))[:majority_count - minority_count]

    # Oversample labels
    y_balanced = np.concatenate([y_train, y_train[oversample_indices]], axis=0)

    # Oversample data per modality
    x_balanced = []
    for i, data in enumerate(x_train_all):
        print(f"\nInput_{i}")
        oversampled_data = data[oversample_indices]
        x_balanced.append(np.concatenate([data, oversampled_data], axis=0))

    return x_balanced, y_balanced


def zeropad_data(data):
    """
    Zero-pad input arrays so that each patient or input type has the same shape.

    Args:
        data: List of arrays [num_samples, height, width].

    Returns:
        List of padded arrays where each is of shape [num_samples * num_inputs, height, width].
    """
    num_inputs = len(data)
    padded_data = []

    for i in range(num_inputs):
        total_samples = data[0].shape[0] * num_inputs
        padded = np.zeros((total_samples, data[i].shape[1], data[i].shape[2]))
        start = data[0].shape[0] * i
        end = data[0].shape[0] * (i + 1)
        padded[start:end] = data[i]
        padded_data.append(padded)

    return padded_data
