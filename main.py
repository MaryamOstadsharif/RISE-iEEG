from utils import *
from MultiPatient_model import MultiPatient_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

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
    'task': 'Singing_Music',
    # number of folds
    'n_folds': 10,
    # set hyperparameter
    'hyper_param': {'F1': 5, 'dropoutRate': 0.542, 'kernLength': 60,
                    'kernLength_sep': 88, 'dropoutType': 'Dropout', 'D': 2},
    'n_ROI': 20,
    'coef_reg': 0.05,
    'epochs': 300,
    'patience': 20,
    'early_stop_monitor': 'val_loss',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    # index of first patient
    'st_num_patient': 0,
    # number of patient for dataset 'audio_visual':51, for dataset 'music_reconstruction':29, for dataset 'move_rest':12
    'num_patient': 29,
    # type_balancing for 'move_rest': 'no_balancing', 'Singing_Music':over_sampling
    'type_balancing': 'over_sampling',
    # Max number of channels for dataset 'audio_visual':164, for dataset 'music_reconstruction':250, dataset 'HTNet':128
    'n_channels_all': 250,
    # use transfer learning, for 'Unseen_patient': True, 'Same_patient': False
    'use_transfer': False,
    # nuber of patients for test in 'Unseen_patient' scanario
    'num_patient_test': 1,
    # path of pretrained model
    'path_save_model': 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2024-02-28-10-14-06/'
                       'accuracy/checkpoint_gen__fold4.h5'
}

paths = Paths(settings)
paths.create_path(path_processed_data=processed_data_path,
                  settings=settings)

model = MultiPatient_model(settings=settings,
                           paths=paths)
model.load_split_data(lp=paths.path_processed_data,
                      n_chans_all=settings['n_channels_all'],
                      type_balancing=settings['type_balancing'])
