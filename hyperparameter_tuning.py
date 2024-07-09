from utils import *
import optuna, joblib, os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use
from MultiPatient_model import MultiPatient_model


def objective(trial):
    print(f'Running Trial {trial.number}')
    params = trial.study.user_attrs
    params['settings'].update({'n_ROI': trial.suggest_int("n_ROI", params['settings']['n_ROI'][0],
                                                          params['settings']['n_ROI'][1],
                                                          step=10)})
    params['settings'].update({'coef_reg': trial.suggest_int("n_ROI", params['settings']['coef_reg'][0],
                                                          params['settings']['coef_reg'][1])})

    params['settings']['hyper_param'].update({'F1': trial.suggest_int("F1", params['settings']['hyper_param']['F1'][0],
                                                      params['settings']['hyper_param']['F1'][1], step=5)})

    params['settings']['hyper_param'].update({'kernLength': trial.suggest_int("kernLength",
                                                          params['settings']['hyper_param']['kernLength'][0],
                                                          params['settings']['hyper_param']['kernLength'][1],
                                                          step=10)})

    params['settings']['hyper_param'].update({'dropoutRate': trial.suggest_float("dropoutRate",
                                              params['settings']['hyper_param']['dropoutRate'][0],
                                              params['settings']['hyper_param']['dropoutRate'][1])})

    params['settings']['hyper_param'].update({'kernLength_sep': trial.suggest_int("kernLength_sep",
                                                  params['settings']['hyper_param']['kernLength_sep'][0],
                                                  params['settings']['hyper_param']['kernLength_sep'][1],
                                                  step=10)})

    params['settings']['hyper_param'].update({'dropoutType': trial.suggest_categorical("dropoutType",
                                                                        params['settings']['hyper_param']['dropoutType'])})

    model = MultiPatient_model(settings=params['settings'],
                               paths=params['paths'])

    return model.load_split_data()


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
    # Task: 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
    'task': 'move_rest',
    # number of folds
    # IF ONE_PATIENT OUT IS tRUE, FOLD=1
    'n_folds': 2,
    # set hyperparameter
    'hyper_param': {'F1': [2, 30], 'dropoutRate': [0.4, 0.8], 'kernLength': [10, 200],
                    'kernLength_sep': [10, 200], 'dropoutType': ['Dropout', 'SpatialDropout2D'], 'D': 2},
    'n_ROI': [5, 100],
    'coef_reg': [0.01, 0.9],
    'epochs': 300,
    'patience': 20,
    'early_stop_monitor': 'val_accuracy',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    # index of first patient
    'st_num_patient': 0,
    # number of patient for dataset 'audio_visual':51, for dataset 'music_reconstruction':29, for dataset 'move_rest':12
    'num_patient': 12,
    'one_patient_out': False,
    # type_balancing for 'move_rest': 'no_balancing', 'Singing_Music':over_sampling
    'type_balancing': 'no_balancing',
    # Max number of channels for dataset 'audio_visual':164, for dataset 'music_reconstruction':250, dataset 'HTNet':128
    'n_channels_all': 128,
    # Whether to use 'Unseen_patient' scenario
    'Unseen_patient': False,
    'del_temporal_lobe': False
}

paths = Paths(settings)
paths.create_base_path(path_processed_data=processed_data_path)
paths.creat_result_path(patient=None)
sp = paths.path_results + 'optuna/'
lp = paths.path_processed_data

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
study.set_user_attr('settings', settings)
study.set_user_attr('paths', paths)
study.optimize(objective, n_trials=30)
joblib.dump(study, sp + 'optuna_study' + '.pkl')
