import os
import pickle as pkl
import numpy as np


def del_temporal_lobe(path, data, task):
    """
    Removes channels associated with the superior temporal lobe for each patient.

    Parameters:
        path (Namespace): Object containing dataset paths (expects `preprocessed_dataset_path` attribute).
        data (list of np.ndarray): List of patient data arrays [num_events, num_channels, num_samples].
        task (str): Task name to locate the appropriate preprocessed dataset.

    Returns:
        list of np.ndarray: Modified data with superior temporal lobe channels removed.
    """
    # Load channel name list
    channel_list_path = os.path.join(path.preprocessed_dataset_path, task, 'channel_name_list.pkl')
    with open(channel_list_path, 'rb') as f:
        channel_names = pkl.load(f)

    # Remove 'superiortemporal' channels from each patient's data
    for patient in range(min(len(data), len(channel_names))):
        del_indices = [
            idx for idx, ch_name in enumerate(channel_names[patient])
            if ch_name[7:] == 'superiortemporal'
        ]
        data[patient] = np.delete(data[patient], del_indices, axis=1)

    return data
