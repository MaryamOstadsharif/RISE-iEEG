import pickle as pkl
import numpy as np

with open('F:/maryam_sh/new_dataset/dataset/data_all_patient_v2.pkl', 'rb') as f:
    data_all_patient = pkl.load(f)

# with open('F:/maryam_sh/new_dataset/dataset/patient_1_reformat.pkl','rb') as f:
#   patient_1_reformat = pkl.load(f)

onset_0 = [14, 16, 24, 26, 28, 34, 36, 43, 45, 47, 56, 57, 64, 66, 73, 75]
onset_1 = [i for i in np.arange(0, 14, 2)] + [19, 21, 30, 32] + [40] + [i for i in np.arange(50, 56, 2)] + \
          [60, 62, 69, 71] + [i for i in np.arange(81, 189, 2)]
onset_1[0] = onset_1[0] + 0.5
onset = onset_1 + onset_0

t_min = 0.5
t_max = 1.5
fs = 100
for patient in range(len(data_all_patient)):
    print('reformat data patient_', str(patient))
    data = data_all_patient[patient].T
    data_reformat = np.zeros((len(onset_1) + len(onset_0), data.shape[0], int((t_min + t_max) * fs)))
    for event in range(len(onset)):
        start_sample = int(onset[event] - t_min) * fs
        end_sample = int(onset[event] + t_max) * fs
        data_reformat[event, :, :] = data[:, start_sample:end_sample]

    with open('F:/maryam_sh/new_dataset/dataset/patient_' + str(patient + 1) + '_reformat.pkl', 'wb') as f:
        pkl.dump(data_reformat, f)

print('end')
