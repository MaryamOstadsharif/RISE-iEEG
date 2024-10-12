from src.data_handling.preprocessing import DataPreprocessor
from src.utils.utils import *
from src.model.MultiPatient_model import MultiPatient_model
from src.settings import Paths, Settings

# Set environment variables to specify which GPU to use
configure_environment()

# Loading Settings from /configs/settings.yaml
settings = Settings()
settings.load_settings()

# Loading Paths from /configs/device.yaml
paths = Paths(settings)
paths.load_device_paths()

# Loading and Preprocessing the Data
data_preprocessor = DataPreprocessor(settings, paths)
data_all_input, labels = data_preprocessor.load_or_preprocess()

if settings.one_patient_out is True:
    for i in range(settings.num_patient):
        # Specify patient to leave out
        settings.del_patient = i
        # Create result path
        paths.update_result_path(patient=i)
        model = MultiPatient_model(settings=settings,
                                   paths=paths)
        model.load_split_data(data_all_input, labels, random_seed=42)

elif settings.unseen_patient:
    for i in range(settings.num_patient):
        settings.st_num_patient = i
        # path of pretrained model
        #TODO: Fix this : Hard coding is not allowd in code
        settings.path_save_model = (('F:/maryam_sh/General model/General '
                                    'code/results/Singing_Music/over_sampling/one_patient_out/') + str(i) +
                                    '/checkpoint_gen__fold0.h5')
        # nuber of patients for test in 'Unseen_patient' scenario
        settings.num_patient_test = 1
        # Create result path
        paths.update_result_path(patient=i)
        model = MultiPatient_model(settings=settings,
                                   paths=paths)
        model.load_split_data(data_all_input, labels, random_seed=42)


else:
    paths.update_result_path(patient=None)
    model = MultiPatient_model(settings=settings,
                               paths=paths)
    model.load_split_data(data_all_input, labels, random_seed=42)
