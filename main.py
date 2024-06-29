from utils import *
from MultiPatient_model import MultiPatient_model

# Set environment variables to specify which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use

# set device
device = 'navid'
if device.lower() == 'navid':
    processed_data_path = 'D:/Datasets/'
elif device.lower() == 'maryam':
    processed_data_path = 'E:/Thesis/new_dataset/'
elif device.lower() == 'system_lab':
    processed_data_path = 'F:/maryam_sh/new_dataset/dataset/'
elif device.lower() == 'navid_lab':
    processed_data_path = ''
else:
    processed_data_path = ''


# Define settings for the model and training process
settings = {
    # Task: 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
    'task': 'move_rest',
    # number of folds
    'n_folds': 10, # Use one patient out cross-validation, set n_folds to 1
    # set hyperparameter
    'hyper_param': {'F1': 5,
                    'dropoutRate': 0.542,
                    'kernLength': 60,
                    'kernLength_sep': 88,
                    'dropoutType': 'Dropout',
                    'D': 2},
    # Number of Regions of Interest
    'n_ROI': 20,
    # Regularization coefficient
    'coef_reg': 0.01,
    # Number of training epochs
    'epochs': 300,
    # Patience for early stopping
    'patience': 20,
    # Metric to monitor for early stopping
    'early_stop_monitor': 'val_accuracy',
    # Optimizer to use
    'optimizer': 'adam',
    # Loss function
    'loss': 'categorical_crossentropy',
    # Index of the first patient
    'st_num_patient': 0,
    # number of patient for dataset 'audio_visual':51, for dataset 'music_reconstruction':29, for dataset 'move_rest':12
    # Number of patients (Audio Visual: 51, Music Reconstruction: 29, Upper-Limb Movement: 12)
    'num_patient': 12,
    # Whether to use one patient out cross-validation
    'one_patient_out': False,
    # # Type of data balancing ('move_rest': 'no_balancing', 'Singing_Music':over_sampling,
    # 'Speech_Music':over_sampling, 'Question_Answer':over_sampling)
    'type_balancing': 'no_balancing',
    # # Max number of channels ('Audio Visual':164, 'Music Reconstruction':250, 'Upper-Limb Movement':128)
    'n_channels_all': 128,
    # Whether to use 'Unseen_patient' scenario
    'Unseen_patient': False,
    'use_transfer': False
}

# Initialize paths
paths = Paths(settings)
paths.create_base_path(path_processed_data=processed_data_path)

if settings['one_patient_out']:
    for i in range(settings['num_patient']):
        # Specify patient to leave out
        settings.update({'del_patient': i})
        # Create result path
        paths.creat_result_path(patient=i)
        model = MultiPatient_model(settings=settings,
                                   paths=paths)
        model.load_split_data()

elif settings['Unseen_patient']:
    for i in range(settings['num_patient']):
        settings.update({'st_num_patient': i})
        # path of pretrained model
        settings.update({'path_save_model': 'F:/maryam_sh/General model/General code/results/Singing_Music/'
                                            'over_sampling/one_patient_out/' + str(i) + '/checkpoint_gen__fold0.h5'})
        # nuber of patients for test in 'Unseen_patient' scenario
        settings.update({'num_patient_test': 1})
        # Create result path
        paths.creat_result_path(patient=i)
        model = MultiPatient_model(settings=settings,
                                   paths=paths)
        model.load_split_data()


else:
    paths.creat_result_path(patient=None)
    model = MultiPatient_model(settings=settings,
                               paths=paths)
    model.load_split_data()
