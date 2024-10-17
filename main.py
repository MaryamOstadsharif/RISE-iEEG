from src.preprocessing.preprocessing import DataPreprocessor
from src.utils.utils import *
from src.model.MultiPatientModel import MultiPatientModel
from src.settings import Paths, Settings
from src.preprocessing import *

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

model = MultiPatientModel(settings=settings,
                          paths=paths)

model.cross_validation(data_all_input, labels, random_seed=42)



