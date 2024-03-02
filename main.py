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
    # 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
    'task': 'Singing_Music',
    'n_folds': 5,
    'hyper_param': {'F1': 5, 'dropoutRate': 0.542, 'kernLength': 60,
                    'kernLength_sep': 88, 'dropoutType': 'Dropout', 'D': 2},
    'epochs': 300,
    'patience': 20,
    'early_stop_monitor': 'val_loss',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    # number of patient for dataset 'audio_visual':51,
    # number of patient for dataset: 'music_reconstruction':29
    # number of patient for dataset: 'move_rest':12
    'num_patient': 20,
    'n_ROI': 20,
    # type_balancing for 'move_rest': 'no_balancing'
    # type_balancing for 'Singing_Music':over_sampling
    'type_balancing': 'over_sampling',
    # number of channels for dataset: 'audio_visual':164,
    # number of channels for dataset: 'music_reconstruction':250
    # number of channels for dataset: 'music_reconstruction':126
    'n_channels_all': 250,
    # use transfer learning, must num_patient= number of all patients
    'use_transfer': True,
    'num_patient_test': 9,
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
