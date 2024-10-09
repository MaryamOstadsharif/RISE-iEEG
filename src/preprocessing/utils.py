import pickle as pkl
import numpy as np


def del_temporal_lobe(path, data, task):
    with open(path.preprocessed_dataset_path + task + '/channel_name_list.pkl', 'rb') as f:
        ch = pkl.load(f)

    for patient in range(29):
        del_ind = []
        for pos, channel in enumerate(ch[patient]):
            if channel[7:] == 'superiortemporal':
                del_ind.append(pos)
        data[patient] = np.delete(data[patient], del_ind, axis=1)

    return data