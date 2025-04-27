import warnings
import time

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from src.model.RISEiEEG_model import *
from src.model.model_utils import *
from sklearn.metrics import roc_curve, roc_auc_score

tf.compat.v1.disable_eager_execution()
from tensorflow.keras import utils as np_utils

os.environ["OMP_NUM_THREADS"] = "1"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import tensorflow as tf
#
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class MultiPatientModel:
    def __init__(self, settings, paths):
        self.path = paths
        self.settings = settings
        self.acc = None
        self.precision = None
        self.recall = None
        self.f1score = None
        self.last_epochs = None
        self.acc_per_patient = None
        self.precision_per_patient = None
        self.recall_per_patient = None
        self.f1score_per_patient = None

    def save_result(self):
        # Save various metrics to numpy files
        np.save(self.path.result_path['All_patients'] + 'acc_RISEiEEG' + '_Fold' + str(self.settings.n_folds) + '.npy', self.acc)
        np.save(self.path.result_path['All_patients'] + 'precision_RISEiEEG' + '_Fold' + str(self.settings.n_folds) + '.npy',
                self.precision)
        np.save(self.path.result_path['All_patients'] + 'recall_RISEiEEG' + '_Fold' + str(self.settings.n_folds) + '.npy', self.recall)
        np.save(self.path.result_path['All_patients'] + 'f1score_RISEiEEG' + '_Fold' + str(self.settings.n_folds) + '.npy', self.f1score)

        # Save per-patient metrics to numpy files
        np.save(self.path.result_path['Seperated_patient'] + 'acc_RISEiEEG_each_patient' + '_Fold' + str(self.settings.n_folds) + '.npy',
                self.acc_per_patient)
        np.save(
            self.path.result_path['Seperated_patient'] + 'precision_RISEiEEG_each_patient' + '_Fold' + str(self.settings.n_folds) + '.npy',
            self.precision_per_patient)
        np.save(self.path.result_path['Seperated_patient'] + 'recall_RISEiEEG_each_patient' + '_Fold' + str(self.settings.n_folds) + '.npy',
                self.recall_per_patient)
        np.save(self.path.result_path['Seperated_patient'] + 'f1score_RISEiEEG_each_patient' + '_Fold' + str(self.settings.n_folds) + '.npy',
                self.f1score_per_patient)

        # Save the last training epochs to a numpy file
        np.save(self.path.path_store_model + 'last_training_epoch' + '_Fold' + str(self.settings.n_folds) + '.npy',
                self.last_epochs)

    def evaluate(self, model, x, y):
        pred = model.predict(x).argmax(axis=-1)
        acc = np.mean(pred == y.argmax(axis=-1))
        pre = precision_recall_fscore_support(y.argmax(axis=-1), pred, average='weighted', zero_division=0)[0]
        recall = precision_recall_fscore_support(y.argmax(axis=-1), pred, average='weighted', zero_division=0)[1]
        f1score = precision_recall_fscore_support(y.argmax(axis=-1), pred, average='weighted', zero_division=0)[2]

        return acc, pre, recall, f1score

    def train(self, X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all, checkpoint_path,
              metrics_compile=['accuracy'], verbos=1, save_best_model=True, mode_regularization='max', batch_size=16):

        if self.settings.mode == 'Unseen_patient':
            num_input = self.settings.num_patient - 1
        else:
            num_input = self.settings.num_patient

        acc_lst = np.zeros(3)
        pre_lst = np.zeros(3)
        recall_lst = np.zeros(3)
        f1score_lst = np.zeros(3)
        if self.settings.mode == 'Same_patient':
            acc_patient = np.zeros(num_input)
            pre_patient = np.zeros(num_input)
            recall_patient = np.zeros(num_input)
            f1score_patient = np.zeros(num_input)
        else:
            acc_patient = np.zeros(self.settings.inner_fold)
            pre_patient = np.zeros(self.settings.inner_fold)
            recall_patient = np.zeros(self.settings.inner_fold)
            f1score_patient = np.zeros(self.settings.inner_fold)

        # Design the model
        model = RISEiEEG(settings=self.settings,
                         nb_classes=len(np.unique(np.argmax(y_train_all, axis=1))),
                         Chans=[X_train_all[i].shape[-1] for i in range(len(X_train_all))],
                         Samples=X_train_all[0].shape[2],
                         num_input=num_input)

        # Compile the model with specified loss function and optimizer
        model.compile(loss=self.settings.loss,
                      optimizer=self.settings.optimizer,
                      metrics=metrics_compile)

        # Set up model checkpointing and early stopping
        checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                                       monitor=self.settings.early_stop_monitor,
                                       mode=mode_regularization,
                                       verbose=verbos,
                                       save_best_only=save_best_model)
        early_stop = EarlyStopping(monitor=self.settings.early_stop_monitor,
                                   mode=mode_regularization,
                                   patience=self.settings.patience,
                                   verbose=verbos)
        # start time for training
        t_start = time.time()

        # Train the model
        model_history = model.fit(X_train_all,
                                  y_train_all,
                                  batch_size=batch_size,
                                  epochs=self.settings.epochs,
                                  verbose=verbos,
                                  validation_data=(X_val_all, y_val_all),
                                  callbacks=[checkpointer, early_stop])

        # Measure training time and determine the last epoch
        t_fit = time.time() - t_start
        last_epoch = len(model_history.history['loss'])
        if last_epoch < self.settings.epochs:
            last_epoch -= self.settings.patience
        print("Last epoch was: ", last_epoch)

        model_history_2s = None
        if self.settings.mode == 'Unseen_patient':
            print('\n ========================= Training model in Second step ==========================')
            pretrained_model = model
            num_input_final = len(X_test_all)
            stage = 'Second_train'
            ind_train_2s, ind_val_2s, ind_test_2s = folds_choose(settings=self.settings,
                                                                 labels=y_test_all,
                                                                 stage=stage,
                                                                 num_folds=self.settings.inner_fold,
                                                                 random_seed=42)
            for fold in range(self.settings.inner_fold):
                print(f' \n ========================= Inner Fold {fold} =========================')
                X_train_2s, y_train_2s, X_val_2s, y_val_2s, X_test_2s, y_test_2s = self.prepare_data(ind_train_2s,
                                                                                                     ind_val_2s,
                                                                                                     ind_test_2s,
                                                                                                     X_test_all,
                                                                                                     y_test_all,
                                                                                                     fold,
                                                                                                     stage='Second_train')

                checkpoint_path_2s = self.path.path_store_model + 'checkpoint_step2_' + '_fold' + str(fold) + '.h5'

                model_final = RISEiEEG(settings=self.settings,
                                       nb_classes=len(np.unique(np.argmax(y_train_2s, axis=1))),
                                       Chans=[X_train_2s[i].shape[-1] for i in range(len(X_train_2s))],
                                       Samples=X_train_2s[0].shape[2],
                                       num_input=num_input_final)

                # Set up model checkpointing and early stopping
                checkpointer_2s = ModelCheckpoint(filepath=checkpoint_path_2s,
                                                  monitor=self.settings.early_stop_monitor,
                                                  mode=mode_regularization,
                                                  verbose=verbos,
                                                  save_best_only=save_best_model)

                early_stop = EarlyStopping(monitor=self.settings.early_stop_monitor,
                                           mode=mode_regularization,
                                           patience=self.settings.patience,
                                           verbose=verbos)

                model_final.compile(loss=self.settings.loss,
                                    optimizer=self.settings.optimizer,
                                    metrics=metrics_compile)

                for i in range(1, 16):
                    model_final.layers[-i].set_weights(pretrained_model.layers[-i].get_weights())
                    model_final.layers[-i].trainable = False

                model_history_2s = model_final.fit(X_train_2s,
                                                   y_train_2s,
                                                   batch_size=batch_size,
                                                   epochs=self.settings.epochs,
                                                   verbose=verbos,
                                                   validation_data=(X_val_2s, y_val_2s),
                                                   callbacks=[checkpointer_2s, early_stop])

                acc_patient[fold], pre_patient[fold], recall_patient[fold], f1score_patient[fold] = (
                    self.evaluate(model_final, X_test_2s, y_test_2s))

        # Evaluate the best model
        model.load_weights(checkpoint_path)

        # Predict and evaluate on the training set
        acc_lst[0], pre_lst[0], recall_lst[0], f1score_lst[0] = self.evaluate(model, X_train_all, y_train_all)

        # Predict and evaluate on the validation set
        acc_lst[1], pre_lst[1], recall_lst[1], f1score_lst[1] = self.evaluate(model, X_val_all, y_val_all)

        # Predict and evaluate on the test set
        if self.settings.mode == 'Same_patient':
            acc_lst[2], pre_lst[2], recall_lst[2], f1score_lst[2] = self.evaluate(model, X_test_all, y_test_all)

        # Evaluate per-patient metrics
        if self.settings.mode == 'Same_patient':
            num_test = X_test_all[0].shape[0] / num_input
            for i in range(num_input):
                x = [X_test_all[k][i * int(num_test):(i + 1) * int(num_test), :, :, :] for k in range(num_input)]
                y = y_test_all[:int(num_test)]
                acc_patient[i], pre_patient[i], recall_patient[i], f1score_patient[i] = self.evaluate(model, x, y)

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test_all[:, 1], model.predict(X_test_all)[:, 1])

            # Compute AUC (Area Under the Curve)
            auc_score = roc_auc_score(y_test_all[:, 1], model.predict(X_test_all)[:, 1])

        # Clear the session to free up resources
        tf.keras.backend.clear_session()
        print(f"\n F1 score:{f1score_patient}")
        return acc_lst, pre_lst, recall_lst, f1score_lst, acc_patient, pre_patient, recall_patient, f1score_patient, \
            model_history.history, fpr, tpr, auc_score

    def prepare_data(self, ind_all_train, ind_all_val, ind_all_test, data_all_input, labels, fold, stage):
        # Get the indices for the current fold
        if self.settings.mode == 'Same_patient' or stage == 'Second_train':
            ind_test = ind_all_test[fold]
        ind_val = ind_all_val[fold]
        ind_train = ind_all_train[fold]

        # Split the data into training, validation, and testing sets
        X_train_all = [data_all_input[i][ind_train] for i in range(len(data_all_input))]
        y_train = labels[ind_train]
        X_val_all = [data_all_input[i][ind_val] for i in range(len(data_all_input))]
        y_val = labels[ind_val]
        if self.settings.mode == 'Same_patient' or stage == 'Second_train':
            X_test_all = [data_all_input[i][ind_test] for i in range(len(data_all_input))]
            y_test = labels[ind_test]
        else:
            X_test_all = [data_all_input[fold]]
            y_test = labels

        # Balance the training data
        if self.settings.type_balancing == 'over_sampling':
            print(' \n ========================= Balancing Data =========================')
            X_train_all, y_train = balance(X_train_all, y_train)

        # Zero-pad the training, validation, and test sets
        X_train_all = zeropad_data(X_train_all)
        X_val_all = zeropad_data(X_val_all)
        y_train_all = y_train.tolist() * len(X_train_all)
        y_val_all = y_val.tolist() * len(X_val_all)
        if self.settings.mode == 'Same_patient' or stage == 'Second_train':
            X_test_all = zeropad_data(X_test_all)
            y_test_all = y_test.tolist() * len(X_test_all)
        else:
            y_test_all = y_test

        # Transpose the data to match the expected input shape
        X_train_all = [np.transpose(X_train_all[i], (0, 2, 1)) for i in range(len(X_train_all))]
        X_val_all = [np.transpose(X_val_all[i], (0, 2, 1)) for i in range(len(X_val_all))]
        if self.settings.mode == 'Same_patient' or stage == 'Second_train':
            X_test_all = [np.transpose(X_test_all[i], (0, 2, 1)) for i in range(len(X_test_all))]

        # Convert labels to categorical format
        # Expand dimensions
        y_train_all = np_utils.to_categorical(y_train_all)
        X_train_all = [np.expand_dims(X_train_all[i], 1) for i in range(len(X_train_all))]
        y_val_all = np_utils.to_categorical(y_val_all)
        X_val_all = [np.expand_dims(X_val_all[i], 1) for i in range(len(X_val_all))]
        if self.settings.mode == 'Same_patient' or stage == 'Second_train':
            y_test_all = np_utils.to_categorical(y_test_all)
            X_test_all = [np.expand_dims(X_test_all[i], 1) for i in range(len(X_test_all))]

        return X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all

    def cross_validation(self, data_all_input, labels, random_seed):
        # Choose indices for training, validation, and testing sets
        if self.settings.mode == 'Same_patient':
            stage = None
        else:
            stage = 'First_train'
        ind_all_train, ind_all_val, ind_all_test = folds_choose(settings=self.settings,
                                                                labels=labels,
                                                                stage=stage,
                                                                num_folds=self.settings.n_folds,
                                                                random_seed=random_seed)

        # Initialize arrays to store accuracy, precision, recall, and fscore for each fold
        self.acc = np.zeros([self.settings.n_folds, 3])
        self.precision = np.zeros([self.settings.n_folds, 3])
        self.recall = np.zeros([self.settings.n_folds, 3])
        self.f1score = np.zeros([self.settings.n_folds, 3])

        # Lists to store per-patient metrics for each fold
        self.acc_per_patient = []
        self.precision_per_patient = []
        self.recall_per_patient = []
        self.f1score_per_patient = []
        for fold in range(self.settings.n_folds):
            print('\n ******************************* Process in Fold ', str(fold), '*******************************')

            X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all = self.prepare_data(ind_all_train,
                                                                                                       ind_all_val,
                                                                                                       ind_all_test,
                                                                                                       data_all_input,
                                                                                                       labels,
                                                                                                       fold,
                                                                                                       stage='First_train')

            # Define the checkpoint path for saving the best model
            checkpoint_path = self.path.path_store_model + 'checkpoint_' + '_fold' + str(fold) + '.h5'

            # Train and evaluate the model
            if self.settings.mode == 'Same_patient':
                print('\n ========================= Training model =========================')
            else:
                print('\n ========================= Training model in First step ==========================')
            acc_lst, pre_lst, recall_lst, f1score_lst, acc_patient, pre_patient, \
                recall_patient, f1score_patient, history, fpr, tpr, auc_score = self.train(X_train_all=X_train_all,
                                                                                           y_train_all=y_train_all,
                                                                                           X_val_all=X_val_all,
                                                                                           y_val_all=y_val_all,
                                                                                           X_test_all=X_test_all,
                                                                                           y_test_all=y_test_all,
                                                                                           checkpoint_path=checkpoint_path)

            # Save the training history for the current fold
            np.save(self.path.path_store_model + 'model_history' + '_Fold' + str(fold) + '.npy', history)
            np.save(self.path.result_path['ROC_curves'] + 'ROC_curve' + '_Fold' + str(fold) + '.npy', [fpr, tpr, auc_score])

            self.acc_per_patient.append(acc_patient)
            self.precision_per_patient.append(pre_patient)
            self.recall_per_patient.append(recall_patient)
            self.f1score_per_patient.append(f1score_patient)

            # Store metrics for the current fold
            for ss in range(3):
                self.acc[fold, ss] = acc_lst[ss]
                self.precision[fold, ss] = pre_lst[ss]
                self.recall[fold, ss] = recall_lst[ss]
                self.f1score[fold, ss] = f1score_lst[ss]

        # Save the overall results
        self.save_result()

        return self.acc[:, 1].mean()
