from utils import *
import optuna, joblib, pdb, os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use
from Multi_patient_test import MultiPatient_model


def objective(trial):
    print(f'Running Trial {trial.number}')
    params = trial.study.user_attrs
    F1 = trial.suggest_int("F1", params['settings']['hyper_param']['F1'][0], params['settings']['hyper_param']['F1'][1])
    model = MultiPatient_model(sp=params["sp"],
                               n_folds=params['settings']['n_folds'],
                               num_patient=params['settings']['num_patient'],
                               n_ROI=trial.suggest_int("n_ROI", params['settings']['n_ROI'][0],
                                                       params['settings']['n_ROI'][1],
                                                       step=10),
                               epochs=params['settings']['epochs'],
                               patience=params['settings']['patience'],
                               F1=F1,
                               dropoutRate=trial.suggest_float("dropoutRate",
                                                               params['settings']['hyper_param']['dropoutRate'][0],
                                                               params['settings']['hyper_param']['dropoutRate'][1]),
                               kernLength=trial.suggest_int("kernLength",
                                                            params['settings']['hyper_param']['kernLength'][0],
                                                            params['settings']['hyper_param']['kernLength'][1],
                                                            step=10),
                               kernLength_sep=trial.suggest_int("kernLength_sep",
                                                                params['settings']['hyper_param']['kernLength_sep'][0],
                                                                params['settings']['hyper_param']['kernLength_sep'][1],
                                                                step=10),
                               dropoutType=trial.suggest_categorical("dropoutType",
                                                                     params['settings']['hyper_param']['dropoutType']),
                               D=params['settings']['hyper_param']['D'],
                               F2=F1 * settings['hyper_param']['D'],
                               loss=params['settings']['loss'],
                               optimizer=params['settings']['optimizer'],
                               task=params['settings']['task'],
                               early_stop_monitor=params['settings']['early_stop_monitor'])

    return model.load_split_data(lp=params["lp"],
                                 n_chans_all=params['settings']['n_channels_all'],
                                 type_balancing=params['settings']['type_balancing'])


# set device
device = 'system_lab'
if device.lower() == 'navid':
    processed_data_path = ''
elif device.lower() == 'maryam':
    processed_data_path = 'E:/Thesis/new_dataset/'
elif device.lower() == 'system_lab':
    processed_data_path = 'F:/maryam_sh/new_dataset/dataset/'
elif device.lower() == 'navid_lab':
    processed_data_path = ''
else:
    processed_data_path = ''

settings = {
    # 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
    'task': 'Singing_Music',
    'n_folds': 2,
    'hyper_param': {'F1': [2, 20], 'dropoutRate': [0.2, 0.8], 'kernLength': [10, 200],
                    'kernLength_sep': [10, 200], 'dropoutType': ['Dropout', 'SpatialDropout2D'], 'D': 2},
    'epochs': 300,
    'patience': 20,
    'early_stop_monitor': 'val_loss',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    # number of patient for dataset 'audio_visual':51,
    # number of patient for dataset: 'music_reconstruction':29
    # number of patient for dataset: 'move_rest':12
    'num_patient': 2,
    'n_ROI': [5, 100],
    # type_balancing for 'move_rest': 'no_balancing'
    'type_balancing': 'over_sampling',
    # number of channels for dataset: 'audio_visual':164,
    # number of channels for dataset: 'music_reconstruction':250
    # number of channels for dataset: 'music_reconstruction':126
    'n_channels_all': 164
}
paths = Paths(settings)
paths.create_path(path_processed_data=processed_data_path,
                  settings=settings)
sp = paths.path_results + 'optuna/'
lp = paths.path_processed_data

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
study.set_user_attr('settings', settings)
study.set_user_attr('sp', sp)
study.set_user_attr('lp', lp)
study.optimize(objective, n_trials=100)
joblib.dump(study, sp + 'optuna_study' + '.pkl')
