from tqdm import tqdm
import numpy as np
import xarray as xr
from collections import Counter
import pickle as pkl

rootpath = 'F:/maryam_sh/HTNet model/'
lp = rootpath + 'ecog_dataset/'
pats_ids_in = ['EC01', 'EC02', 'EC03', 'EC04', 'EC05', 'EC06', 'EC07', 'EC08', 'EC09', 'EC10', 'EC11', 'EC12']
n_chans_all = 126
tlim = [-1, 1]
event_types = ['rest', 'move']

if not isinstance(pats_ids_in, list):
    pats_ids_in = [pats_ids_in]

# Gather each subjects data, and concatenate all days
for j in tqdm(range(len(pats_ids_in))):
    pat_curr = pats_ids_in[j]
    ep_data_in = xr.open_dataset(lp + pat_curr + '_ecog_data.nc')
    ep_times = np.asarray(ep_data_in.time)
    time_inds = np.nonzero(np.logical_and(ep_times >= tlim[0], ep_times <= tlim[1]))[0]
    n_ecog_chans = (len(ep_data_in.channels) - 1)

    n_chans_curr = n_ecog_chans

    days_all_in = np.asarray(ep_data_in.events)
    days_train_inds = []
    for day_tmp in list(np.unique(days_all_in)):
        days_train_inds.extend(np.nonzero(days_all_in == day_tmp)[0])

    dat = ep_data_in[dict(events=days_train_inds, channels=slice(0, n_chans_curr),
                          time=time_inds)].to_array().values.squeeze()
    labels = ep_data_in[dict(events=days_train_inds, channels=ep_data_in.channels[-1],
                             time=0)].to_array().values.squeeze()
    labels = labels - 1


    data_reformat = np.zeros((300,dat.shape[1],dat.shape[2]))
    n = 0
    for k in range(dat.shape[0]):
        if labels[k] == 0 and n < 150:
            data_reformat[n, :, :] = dat[k, :, :]
            n = n + 1

    for k in range(dat.shape[0]):
        if labels[k] == 1 and n < 300:
            data_reformat[n, :, :] = dat[k, :, :]
            n = n + 1


    with open('F:/maryam_sh/new_dataset/dataset/move_rest/patient_' + str(j + 1) + '_reformat.pkl', 'wb') as f:
        pkl.dump(data_reformat, f)

label = np.array([0] * 150 + [1] * 150)
with open('F:/maryam_sh/new_dataset/dataset/move_rest/labels.pkl', 'wb') as f:
    pkl.dump(label, f)
