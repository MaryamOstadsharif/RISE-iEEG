
import pandas as pd
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

def read_time(task, t_min, paths):
    if task == 'Question_Answer':
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

    if task == 'Speech_Music':
        onset_1 = [i for i in np.arange(0, 390, 60)]
        offset_1 = [i for i in np.arange(30, 420, 60)]
        onset_0 = [i for i in np.arange(30, 390, 60)]
        offset_0 = [i for i in np.arange(60, 390, 60)]
        onset_1[0] = onset_1[0] + t_min

    return onset_1, offset_1, onset_0, offset_0