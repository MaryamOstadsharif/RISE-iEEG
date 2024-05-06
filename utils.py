import os
import json
from pathlib import Path
import datetime
import numpy as np
from collections import Counter
import pandas as pd


class Paths:
    def __init__(self, settings):
        self.settings = settings
        self.path_processed_data = ''
        self.pro_mat_path = ''
        self.path_store_best_model = ''
        self.path_best_result = ''
        self.settings = settings
        self.path_results = ''
        self.path_error_analysis = ''
        self.path_store_model = ''
        self.path_save_data = ''

    def create_path(self, path_processed_data, settings):
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
        base_path = dir_path + '/results/' + settings['task'] + '/' + settings['type_balancing'] + '/' + \
                    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'
        # base_path = dir_path + '/results/' + settings['task'] + '/' + settings['type_balancing'] + '/' + \
        #             '/train_test_each_patient'+'/' +str(i) + '/'

        # Place where we save preprocessed data
        self.path_processed_data = path_processed_data
        # Place where we save features
        self.path_save_data = dir_path + '/data/'
        Path(self.path_save_data).mkdir(parents=True, exist_ok=True)
        # Place where we save figures
        self.path_results = base_path + 'accuracy/'
        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        # Place where we save model hyper parameters
        self.path_store_model = base_path + 'hyper_param_set/saved_models/'
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        self.save_settings()

    def save_settings(self):
        """ working directory """
        """ write settings to the json file """
        with open(self.path_store_model + 'settings.json', 'w') as fp:
            json.dump(self.settings, fp)
