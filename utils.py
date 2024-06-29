import os
import json
from pathlib import Path
import datetime


class Paths:
    def __init__(self, settings):
        self.settings = settings
        self.path_processed_data = ''
        self.settings = settings
        self.path_results = ''
        self.path_store_model = ''
        self.base_path = ''

    def create_base_path(self, path_processed_data):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :param path_dataset:
        :param path_processed_data:
        :returns:
            path_results: the path for saving results'
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.base_path = dir_path + '/results/' + self.settings['task'] + '/' + self.settings['type_balancing'] + '/' + \
                    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'
        # Place where we save preprocessed data
        self.path_processed_data = path_processed_data

    def creat_result_path(self, patient):
        if self.settings['one_patient_out'] or self.settings['Unseen_patient']:
            # Place where we save figures
            self.path_results = self.base_path + str(patient) + '/accuracy/'
            Path(self.path_results).mkdir(parents=True, exist_ok=True)
            # Place where we save model hyper parameters
            self.path_store_model = self.base_path + str(patient) + '/hyper_param_set/saved_models/'
            Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        else:
            self.path_results = self.base_path + 'accuracy/'
            Path(self.path_results).mkdir(parents=True, exist_ok=True)
            # Place where we save model hyper parameters
            self.path_store_model = self.base_path + 'hyper_param_set/saved_models/'
            Path(self.path_store_model).mkdir(parents=True, exist_ok=True)

        self.save_settings()

    def save_settings(self):
        """ working directory """
        """ write settings to the json file """
        with open(self.path_store_model + 'settings.json', 'w') as fp:
            json.dump(self.settings, fp)
