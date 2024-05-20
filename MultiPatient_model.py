import warnings
import time
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import numpy as np
from ECoGNet_model import *
from model_utils import *

tf.compat.v1.disable_eager_execution()
from tensorflow.keras import utils as np_utils

rand_seed = 1337
# tf.random.set_seed(rand_seed)


os.environ["OMP_NUM_THREADS"] = "1"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class MultiPatient_model:
    def __init__(self, settings, paths):
        self.path = paths
        self.settings = settings

    def save_result(self):
        # Save various metrics to numpy files
        np.save(self.path.path_results + 'acc_ECoGNet' + '_' + str(self.settings['n_folds']) + '.npy', self.accs)
        np.save(self.path.path_results + 'precision_ECoGNet' + '_' + str(self.settings['n_folds']) + '.npy',
                self.precision)
        np.save(self.path.path_results + 'recall_ECoGNet' + '_' + str(self.settings['n_folds']) + '.npy', self.recall)
        np.save(self.path.path_results + 'fscore_ECoGNet' + '_' + str(self.settings['n_folds']) + '.npy', self.fscore)

        # Save per-patient metrics to numpy files
        np.save(self.path.path_results + 'acc_ECoGNet_each_patient' + '_' + str(self.settings['n_folds']) + '.npy',
                self.acc_patient_folds)
        np.save(
            self.path.path_results + 'precision_ECoGNet_each_patient' + '_' + str(self.settings['n_folds']) + '.npy',
            self.precision_patient_folds)
        np.save(self.path.path_results + 'recall_ECoGNet_each_patient' + '_' + str(self.settings['n_folds']) + '.npy',
                self.recall_patient_folds)
        np.save(self.path.path_results + 'fscore_ECoGNet_each_patient' + '_' + str(self.settings['n_folds']) + '.npy',
                self.fscore_patient_folds)

        # Save the last training epochs to a numpy file
        np.save(self.path.path_results + 'last_training_epoch' + '_' + str(self.settings['n_folds']) + '.npy',
                self.last_epochs)

    def train_evaluate_model(self, X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all, chckpt_path):

        if self.settings['Unseen_patient']:
            num_input = self.settings['num_patient_test']
            pretrained_model = tf.keras.models.load_model(self.settings['path_save_model'])
        else:
            if self.settings['one_patient_out']:
                num_input = self.settings['num_patient'] - 1
            else:
                num_input = self.settings['num_patient']

        # Design the model
        model = ECoGNet(settings=self.settings,
                        nb_classes=len(np.unique(np.argmax(y_train_all, axis=1))),
                        Chans=[X_train_all[i].shape[-1] for i in range(len(X_train_all))],
                        Samples=X_train_all[0].shape[2],
                        num_input=num_input)

        # Compile the model with specified loss function and optimizer
        model.compile(loss=self.settings['loss'],
                      optimizer=self.settings['optimizer'],
                      metrics=['accuracy'])

        # Set up model checkpointing and early stopping
        checkpointer = ModelCheckpoint(filepath=chckpt_path,
                                       monitor='val_accuracy',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True)
        early_stop = EarlyStopping(monitor=self.settings['early_stop_monitor'],
                                   mode='max',
                                   patience=self.settings['patience'],
                                   verbose=0)
        # start time for training
        t_start = time.time()

        # Freeze the last few layers of the pretrained model
        if self.settings['Unseen_patient']:
            for i in range(1, 16):
                model.layers[-i].set_weights(pretrained_model.layers[-i].get_weights())
                model.layers[-i].trainable = False

        # Train the model
        model_history = model.fit(X_train_all,
                                  y_train_all,
                                  batch_size=16,
                                  epochs=self.settings['epochs'],
                                  verbose=2,
                                  validation_data=(X_val_all, y_val_all),
                                  callbacks=[checkpointer, early_stop])

        # Measure training time and determine the last epoch
        t_fit = time.time() - t_start
        last_epoch = len(model_history.history['loss'])
        if last_epoch < self.settings['epochs']:
            last_epoch -= self.settings['patience']
        print("Last epoch was: ", last_epoch)

        # Evaluate the best model
        model.load_weights(chckpt_path)
        accs_lst = []
        pre_lst = []
        recall_lst = []
        fscore_lst = []
        acc_patient = []
        pre_patient = []
        recall_patient = []
        fscore_patinet = []

        # Predict and evaluate on the training set
        preds = model.predict(X_train_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_train_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

        # Predict and evaluate on the validation set
        preds = model.predict(X_val_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_val_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

        # Predict and evaluate on the test set
        preds = model.predict(X_test_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_test_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

        # Evaluate per-patient metrics
        num_test = X_test_all[0].shape[0] / num_input
        for i in range(num_input):
            preds = model.predict(
                [X_test_all[k][i * int(num_test):(i + 1) * int(num_test), :, :, :] for k in
                 range(num_input)]).argmax(
                axis=-1)
            acc_patient.append(np.mean(preds == y_test_all[:int(num_test)].argmax(axis=-1)))
            pre_patient.append(
                precision_recall_fscore_support(y_test_all[:int(num_test)].argmax(axis=-1), preds, average='weighted',
                                                zero_division=0)[0])
            recall_patient.append(
                precision_recall_fscore_support(y_test_all[:int(num_test)].argmax(axis=-1), preds, average='weighted',
                                                zero_division=0)[1])
            fscore_patinet.append(
                precision_recall_fscore_support(y_test_all[:int(num_test)].argmax(axis=-1), preds, average='weighted',
                                                zero_division=0)[2])

        # Clear the session to free up resources
        tf.keras.backend.clear_session()
        print(f"\n F1 score:{fscore_lst[2]}")
        return accs_lst, pre_lst, recall_lst, fscore_lst, acc_patient, pre_patient, recall_patient, fscore_patinet, \
               np.array([last_epoch, t_fit]), model_history.history

    def load_split_data(self):
        # Set the random seed
        np.random.seed(rand_seed)

        # Load the input data and labels
        data_all_input, labels = load_data(path=self.path,
                                           settings=self.settings)

        # Choose indices for training, validation, and testing sets
        inds_all_train, inds_all_val, inds_all_test = folds_choose(settings=self.settings,
                                                                   labels=labels,
                                                                   num_events=data_all_input[0].shape[0],
                                                                   num_minority=Counter(labels).most_common()[1][1],
                                                                   num_majority=Counter(labels).most_common()[0][1],
                                                                   random_seed=rand_seed)

        # Initialize arrays to store accuracy, precision, recall, and fscore for each fold
        self.accs = np.zeros([self.settings['n_folds'], 3])
        self.precision = np.zeros([self.settings['n_folds'], 3])
        self.recall = np.zeros([self.settings['n_folds'], 3])
        self.fscore = np.zeros([self.settings['n_folds'], 3])
        self.last_epochs = np.zeros([self.settings['n_folds'], 2])

        # Lists to store per-patient metrics for each fold
        self.acc_patient_folds = []
        self.precision_patient_folds = []
        self.recall_patient_folds = []
        self.fscore_patient_folds = []
        for fold in range(self.settings['n_folds']):
            print('\n ******************************* procsess in fold ', str(fold), '*******************************')

            # Get the indices for the current fold
            inds_test = inds_all_test[fold]
            inds_val = inds_all_val[fold]
            inds_train = inds_all_train[fold]

            # Split the data into training, validation, and testing sets
            X_train_all = ([data_all_input[i][inds_train] for i in range(len(data_all_input))])
            y_train = labels[inds_train]
            X_test_all = ([data_all_input[i][inds_test] for i in range(len(data_all_input))])
            y_test = labels[inds_test]
            X_val_all = ([data_all_input[i][inds_val] for i in range(len(data_all_input))])
            y_val = labels[inds_val]

            # Balance the training data
            if self.settings['type_balancing'] == 'over_sampling':
                print(' \n ========================= Balancing Data =========================')
                X_train_all, y_train = balance(X_train_all, y_train)

            # Zero-pad the training, validation, and test sets
            X_train_all, X_test_all, X_val_all = zeropad_data(X_train_all, X_test_all, X_val_all, len(data_all_input))
            y_train_all = y_train.tolist() * len(data_all_input)
            y_test_all = y_test.tolist() * len(data_all_input)
            y_val_all = y_val.tolist() * len(data_all_input)

            # Transpose the data to match the expected input shape
            X_train_all = [np.transpose(X_train_all[i], (0, 2, 1)) for i in range(len(X_train_all))]
            X_val_all = [np.transpose(X_val_all[i], (0, 2, 1)) for i in range(len(X_val_all))]
            X_test_all = [np.transpose(X_test_all[i], (0, 2, 1)) for i in range(len(X_test_all))]

            # Convert labels to categorical format
            # Expand dimensions
            y_train_all = np_utils.to_categorical(y_train_all)
            X_train_all = [np.expand_dims(X_train_all[i], 1) for i in range(len(X_train_all))]
            y_val_all = np_utils.to_categorical(y_val_all)
            X_val_all = [np.expand_dims(X_val_all[i], 1) for i in range(len(X_val_all))]
            y_test_all = np_utils.to_categorical(y_test_all)
            X_test_all = [np.expand_dims(X_test_all[i], 1) for i in range(len(X_test_all))]

            # Define the checkpoint path for saving the best model
            chckpt_path = self.path.path_results + 'checkpoint_gen_' + '_fold' + str(fold) + '.h5'

            # Train and evaluate the model
            print('\n ========================= Training model =========================')
            accs_lst, pre_lst, recall_lst, fscore_lst, acc_patient, pre_patient, \
            recall_patient, fscore_patine, last_epoch_tmp, history = self.train_evaluate_model(X_train_all=X_train_all,
                                                                                               y_train_all=y_train_all,
                                                                                               X_val_all=X_val_all,
                                                                                               y_val_all=y_val_all,
                                                                                               X_test_all=X_test_all,
                                                                                               y_test_all=y_test_all,
                                                                                               chckpt_path=chckpt_path)

            # Save the training history for the current fold
            np.save(self.path.path_results + 'model_history' + '_' + str(fold) + '.npy', history)

            self.acc_patient_folds.append(acc_patient)
            self.precision_patient_folds.append(pre_patient)
            self.recall_patient_folds.append(recall_patient)
            self.fscore_patient_folds.append(fscore_patine)

            # Store metrics for the current fold
            for ss in range(3):
                self.accs[fold, ss] = accs_lst[ss]
                self.precision[fold, ss] = pre_lst[ss]
                self.recall[fold, ss] = recall_lst[ss]
                self.fscore[fold, ss] = fscore_lst[ss]

            # Store the last epoch for the current fold
            self.last_epochs[fold, :] = last_epoch_tmp

        # Save the overall results
        self.save_result()

        return self.accs[:, 1].mean()
