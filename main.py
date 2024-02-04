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
    # 'Question_Answer' & 'Singing_Music' & 'Speech_Music'
    'task': 'Singing_Music',
    'n_folds': 29,
    'hyper_param': {'F1': 5, 'dropoutRate': 0.542, 'kernLength': 60,
                    'kernLength_sep': 88, 'dropoutType': 'Dropout', 'D': 2},
    'epochs': 300,
    'patience': 20,
    'early_stop_monitor': 'val_loss',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    # number of patient for dataset: 'audio_visual':51, for dataset: 'music_reconstruction':29
    'num_patient': 29,
    'n_ROI': 20,
    'type_balancing': 'over_sampling',
    # number of patient for dataset: 'audio_visual':164, for dataset: 'music_reconstruction':250
    'n_channels_all': 250
}
paths = Paths(settings)
paths.create_path(path_processed_data=processed_data_path,
                  settings=settings)

model=MultiPatient_model(sp=paths.path_results,
                         n_folds=settings['n_folds'],
                         num_patient=settings['num_patient'],
                         n_ROI=settings['n_ROI'],
                         epochs=settings['epochs'],
                         patience=settings['patience'],
                         F1=settings['hyper_param']['F1'],
                         dropoutRate=settings['hyper_param']['dropoutRate'],
                         kernLength=settings['hyper_param']['kernLength'],
                         kernLength_sep=settings['hyper_param']['kernLength_sep'],
                         dropoutType=settings['hyper_param']['dropoutType'],
                         D=settings['hyper_param']['D'],
                         F2=settings['hyper_param']['F1'] * settings['hyper_param']['D'],
                         loss=settings['loss'],
                         optimizer=settings['optimizer'],
                         task=settings['task'],
                         early_stop_monitor=settings['early_stop_monitor'])

model.load_split_data(lp=paths.path_processed_data,
                       n_chans_all=settings['n_channels_all'],
                       type_balancing=settings['type_balancing'])