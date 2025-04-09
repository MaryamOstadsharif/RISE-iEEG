from pathlib import Path
import yaml
import os
import warnings


class Settings:
    def __init__(self):
        self.__supported_mode = ['Same_patient', 'Unseen_patient']
        self.__supported_tasks = ['Singing_Music', 'Move_Rest']
        self.__supported_optimizers = ['adam', 'sgd', 'rmsprop']
        self.__supported_losses = ['categorical_crossentropy', 'binary_crossentropy', 'mse']
        self.__supported_dropout_types = ['Dropout', 'AlphaDropout', 'SpatialDropout2D']
        self.__mode = None
        self.__task = None
        self.__load_preprocessed_data = None
        self.__num_patient = None
        self.__n_ROI = None
        self.__coef_reg = None
        self.__n_folds = None
        self.__inner_fold = None
        self.__type_balancing = None
        self.__debug_mode = None
        self.__epochs = None
        self.__patience = None
        self.__early_stop_monitor = None
        self.__optimizer = None
        self.__loss = None
        self.__del_temporal_lobe = None
        self.__hyper_param = {'F1': 5,
                              'dropoutRate': 0.542,
                              'kernLength': 60,
                              'kernLength_sep': 88,
                              'dropoutType': 'Dropout',
                              'D': 2}

        self.del_patient = None

    def load_settings(self):
        """
        This function loads the YAML files for settings and network settings from the working directory and
        creates a Settings object based on the fields in the YAML file. It also loads the local path of the datasets
        from device_path.yaml
        return:
            settings: a Settings object
            network_settings: a dictionary containing settings of the saved_model
            device_path: the path to the datasets on the local device
        """

        """ working directory """
        working_folder = Path(__file__).resolve().parents[2]
        config_folder = working_folder / 'configs'

        """ loading settings from the yaml file """
        try:
            with open(config_folder / "settings.yaml", "r") as file:
                settings_yaml = yaml.safe_load(file)
        except Exception as e:
            raise Exception('Could not load settings.yaml from the working directory!') from e

        for key, value in settings_yaml.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

    def save_settings(self, file_path):
        """
        Save all the attributes of the Settings class to a YAML file.

        :param file_path: Path where the YAML file will be saved.
        """
        settings_dict = {
            'mode': self.__mode,
            'task': self.__task,
            'load_preprocessed_data': self.__load_preprocessed_data,
            'num_patient': self.__num_patient,
            'n_ROI': self.__n_ROI,
            'coef_reg': self.__coef_reg,
            'n_folds': self.__n_folds,
            'inner_fold': self.__inner_fold,
            'type_balancing': self.__type_balancing,
            'debug_mode': self.__debug_mode,
            'epochs': self.__epochs,
            'patience': self.__patience,
            'early_stop_monitor': self.__early_stop_monitor,
            'optimizer': self.__optimizer,
            'loss': self.__loss,
            'del_temporal_lobe': self.__del_temporal_lobe,
            'hyper_param': self.__hyper_param
        }

        # with open(file_path, 'w') as yaml_file:
        #     yaml.dump(settings_dict, yaml_file, default_flow_style=False)

    @property
    def hyper_param(self):
        return self.__hyper_param

    @hyper_param.setter
    def hyper_param(self, value):
        required_keys = ['F1', 'dropoutRate', 'kernLength', 'kernLength_sep', 'dropoutType', 'D']

        if not isinstance(value, dict):
            raise ValueError("hyper_param must be a dictionary")

        for key in required_keys:
            if key not in value:
                raise ValueError(f"Missing required key in hyper_param: {key}")

        if not isinstance(value['F1'], int) or value['F1'] <= 0:
            raise ValueError("F1 must be a positive integer")

        if not isinstance(value['dropoutRate'], (float, int)) or not (0 <= value['dropoutRate'] <= 1):
            raise ValueError("dropoutRate must be a float between 0 and 1")

        if not isinstance(value['kernLength'], int) or value['kernLength'] <= 0:
            raise ValueError("kernLength must be a positive integer")

        if not isinstance(value['kernLength_sep'], int) or value['kernLength_sep'] <= 0:
            raise ValueError("kernLength_sep must be a positive integer")

        if value['dropoutType'] not in self.__supported_dropout_types:
            raise ValueError(f"dropoutType must be one of {self.__supported_dropout_types}")

        if not isinstance(value['D'], int) or value['D'] <= 0:
            raise ValueError("D must be a positive integer")

        self.__hyper_param = value

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, value):
        if value in self.__supported_mode:
            self.__mode = value
        else:
            raise ValueError(f"Task should be one of {self.__supported_mode}")

    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, value):
        if value in self.__supported_tasks:
            self.__task = value
        else:
            raise ValueError(f"Task should be one of {self.__supported_tasks}")

    @property
    def load_preprocessed_data(self):
        return self.__load_preprocessed_data

    @load_preprocessed_data.setter
    def load_preprocessed_data(self, value):
        if isinstance(value, bool):
            self.__load_preprocessed_data = value
        else:
            raise ValueError("load_preprocessed_data must be a boolean")


    @property
    def num_patient(self):
        return self.__num_patient

    @num_patient.setter
    def num_patient(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_patient = value
        else:
            raise ValueError("num_patient must be a positive integer")


    @property
    def n_ROI(self):
        return self.__n_ROI

    @n_ROI.setter
    def n_ROI(self, value):
        if isinstance(value, int) and value > 0:
            self.__n_ROI = value
        else:
            raise ValueError("n_ROI must be a positive integer")

    @property
    def coef_reg(self):
        return self.__coef_reg

    @coef_reg.setter
    def coef_reg(self, value):
        if isinstance(value, (float, int)) and value >= 0:
            self.__coef_reg = value
        else:
            raise ValueError("coef_reg must be a non-negative number")

    @property
    def n_folds(self):
        return self.__n_folds

    @n_folds.setter
    def n_folds(self, value):
        if isinstance(value, int) and value > 0:
            self.__n_folds = value
        else:
            raise ValueError("n_folds must be a positive integer")

    @property
    def inner_fold(self):
        return self.__inner_fold

    @inner_fold.setter
    def inner_fold(self, value):
        if isinstance(value, int) and value > 0:
            self.__inner_fold = value
        else:
            raise ValueError("n_folds must be a positive integer")


    @property
    def type_balancing(self):
        return self.__type_balancing

    @type_balancing.setter
    def type_balancing(self, value):
        if isinstance(value, str):
            self.__type_balancing = value
        else:
            raise ValueError("type_balancing must be a string")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("debug_mode must be a boolean")

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, value):
        if isinstance(value, int) and value > 0:
            self.__epochs = value
        else:
            raise ValueError("epochs must be a positive integer")

    @property
    def patience(self):
        return self.__patience

    @patience.setter
    def patience(self, value):
        if isinstance(value, int) and value > 0:
            self.__patience = value
        else:
            raise ValueError("patience must be a positive integer")

    @property
    def early_stop_monitor(self):
        return self.__early_stop_monitor

    @early_stop_monitor.setter
    def early_stop_monitor(self, value):
        if isinstance(value, str):
            self.__early_stop_monitor = value
        else:
            raise ValueError("early_stop_monitor must be a string")

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, value):
        if value in self.__supported_optimizers:
            self.__optimizer = value
        else:
            raise ValueError(f"optimizer should be one of {self.__supported_optimizers}")

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        if value in self.__supported_losses:
            self.__loss = value
        else:
            raise ValueError(f"loss should be one of {self.__supported_losses}")

    @property
    def del_temporal_lobe(self):
        return self.__del_temporal_lobe

    @del_temporal_lobe.setter
    def del_temporal_lobe(self, value):
        if isinstance(value, bool):
            self.__del_temporal_lobe = value
        else:
            raise ValueError("del_temporal_lobe must be a boolean")
