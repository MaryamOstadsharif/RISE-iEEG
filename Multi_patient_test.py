import warnings
import os, time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import numpy as np
from tqdm import tqdm
from ECoGNet_model import *
from model_utils import *

os.environ["OMP_NUM_THREADS"] = "1"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class MultiPatient_model:
    def __init__(self, sp, num_patient, n_ROI, task, dropoutRate, kernLength, F1, D, F2, dropoutType,
                 kernLength_sep, loss, optimizer, patience, early_stop_monitor, epochs, n_folds):
        self.sp = sp
        self.num_patient = num_patient
        self.n_ROI = n_ROI
        self.task = task
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType
        self.kernLength_sep = kernLength_sep
        self.loss = loss
        self.optimizer = optimizer
        self.patience = patience
        self.early_stop_monitor = early_stop_monitor
        self.epochs = epochs
        self.n_folds = n_folds
        self.use_transfer = False
        self.num_patient_test = 0
        self.path_save_model = ''

    def save_result(self):

        np.save(self.sp + 'acc_GNCNN' + '_' + str(self.n_folds) + '.npy', self.accs)
        np.save(self.sp + 'precision_GNCNN' + '_' + str(self.n_folds) + '.npy', self.precision)
        np.save(self.sp + 'recall_GNCNN' + '_' + str(self.n_folds) + '.npy', self.recall)
        np.save(self.sp + 'fscore_GNCNN' + '_' + str(self.n_folds) + '.npy', self.fscore)

        np.save(self.sp + 'acc_GNCNN_each_patient' + '_' + str(self.n_folds) + '.npy', self.acc_patient_folds)
        np.save(self.sp + 'precision_GNCNN_each_patient' + '_' + str(self.n_folds) + '.npy',
                self.precision_patient_folds)
        np.save(self.sp + 'recall_GNCNN_each_patient' + '_' + str(self.n_folds) + '.npy', self.recall_patient_folds)
        np.save(self.sp + 'fscore_GNCNN_each_patient' + '_' + str(self.n_folds) + '.npy', self.fscore_patient_folds)

        np.save(self.sp + 'last_training_epoch' + '_' + str(self.n_folds) + '.npy', self.last_epochs)

    def train_evaluate_model(self, X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all, chckpt_path):

        if self.use_transfer:
            num_input = self.num_patient_test
        else:
            num_input = self.num_patient
        # design GNCNN model
        model = ECoGNet(nb_classes=len(np.unique(np.argmax(y_train_all, axis=1))),
                        Chans=X_train_all[0].shape[-1],
                        Samples=X_train_all[0].shape[2],
                        dropoutRate=self.dropoutRate,
                        n_ROI=self.n_ROI, kernLength=self.kernLength,
                        F1=self.F1, D=self.D, F2=self.F2,
                        dropoutType=self.dropoutType,
                        kernLength_sep=self.kernLength_sep,
                        num_input=num_input,
                        use_transfer=False,
                        num_input_pretrained_model=0)

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=chckpt_path, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor=self.early_stop_monitor, mode='min', patience=self.patience, verbose=0)
        t_start = time.time()

        if self.use_transfer:
            pretrained_model = tf.keras.models.load_model(self.path_save_model)
            for i in range(1, 15):
                model.layers[-i].set_weights(pretrained_model.layers[-i].get_weights())
                model.layers[-i].trainable = False

        # training GNCNN model
        model_history = model.fit(X_train_all, y_train_all, batch_size=16, epochs=self.epochs, verbose=2,
                                  validation_data=(X_val_all, y_val_all), callbacks=[checkpointer, early_stop])
        t_fit = time.time() - t_start
        last_epoch = len(model_history.history['loss'])
        if last_epoch < self.epochs:
            last_epoch -= self.patience
        print("Last epoch was: ", last_epoch)

        # evaluate the best model
        model.load_weights(chckpt_path)
        accs_lst = []
        pre_lst = []
        recall_lst = []
        fscore_lst = []
        acc_patient = []
        pre_patient = []
        recall_patient = []
        fscore_patinet = []

        preds = model.predict(X_train_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_train_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_train_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

        preds = model.predict(X_val_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_val_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_val_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

        preds = model.predict(X_test_all).argmax(axis=-1)
        accs_lst.append(np.mean(preds == y_test_all.argmax(axis=-1)))
        pre_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[0])
        recall_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[1])
        fscore_lst.append(
            precision_recall_fscore_support(y_test_all.argmax(axis=-1), preds, average='weighted', zero_division=0)[2])

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

        tf.keras.backend.clear_session()
        return accs_lst, pre_lst, recall_lst, fscore_lst, np.array(
            [last_epoch, t_fit]), acc_patient, pre_patient, recall_patient, fscore_patinet

    def load_split_data(self, lp, n_chans_all, type_balancing, rand_seed=1337):
        np.random.seed(rand_seed)

        data_all_input, labels = load_data(self.num_patient,
                                           lp,
                                           n_chans_all=n_chans_all,
                                           task=self.task,
                                           use_transfer=self.use_transfer,
                                           num_patient_test=self.num_patient_test)
        if self.task != 'Singing_Music':
            labels = np.array(labels)

        inds_all_train, inds_all_val, inds_all_test = folds_choose(self.n_folds,
                                                                   labels,
                                                                   data_all_input[0].shape[0],
                                                                   Counter(labels).most_common()[1][1],
                                                                   Counter(labels).most_common()[0][1],
                                                                   type_balancing,
                                                                   self.task,
                                                                   rand_seed)

        self.accs = np.zeros([self.n_folds, 3])
        self.precision = np.zeros([self.n_folds, 3])
        self.recall = np.zeros([self.n_folds, 3])
        self.fscore = np.zeros([self.n_folds, 3])
        self.last_epochs = np.zeros([self.n_folds, 2])

        self.acc_patient_folds = []
        self.precision_patient_folds = []
        self.recall_patient_folds = []
        self.fscore_patient_folds = []
        for fold in tqdm(range(self.n_folds)):
            print('procsess in fold_', str(fold))
            inds_test = inds_all_test[fold]
            inds_val = inds_all_val[fold]
            inds_train = inds_all_train[fold]

            X_train_all = ([data_all_input[i][inds_train] for i in range(len(data_all_input))])
            y_train = labels[inds_train]
            X_test_all = ([data_all_input[i][inds_test] for i in range(len(data_all_input))])
            y_test = labels[inds_test]
            X_val_all = ([data_all_input[i][inds_val] for i in range(len(data_all_input))])
            y_val = labels[inds_val]

            if type_balancing == 'over_sampling':
                X_train_all, y_train = balance(X_train_all, y_train)

            y_train_all = y_train.tolist() * len(data_all_input)
            X_train_all, X_test_all, X_val_all = zeropad_data(X_train_all, X_test_all, X_val_all, len(data_all_input))
            y_test_all = y_test.tolist() * len(data_all_input)
            y_val_all = y_val.tolist() * len(data_all_input)

            X_train_all = [np.transpose(X_train_all[i], (0, 2, 1)) for i in range(len(X_train_all))]
            X_val_all = [np.transpose(X_val_all[i], (0, 2, 1)) for i in range(len(X_val_all))]
            X_test_all = [np.transpose(X_test_all[i], (0, 2, 1)) for i in range(len(X_test_all))]

            y_train_all = np_utils.to_categorical(y_train_all)
            X_train_all = [np.expand_dims(X_train_all[i], 1) for i in range(len(X_train_all))]
            y_val_all = np_utils.to_categorical(y_val_all)
            X_val_all = [np.expand_dims(X_val_all[i], 1) for i in range(len(X_val_all))]
            y_test_all = np_utils.to_categorical(y_test_all)
            X_test_all = [np.expand_dims(X_test_all[i], 1) for i in range(len(X_test_all))]

            chckpt_path = self.sp + 'checkpoint_gen_' + '_fold' + str(fold) + '.h5'

            accs_lst, pre_lst, recall_lst, fscore_lst, last_epoch_tmp, acc_patient, pre_patient, \
            recall_patient, fscore_patine = self.train_evaluate_model(X_train_all=X_train_all,
                                                                      y_train_all=y_train_all,
                                                                      X_val_all=X_val_all,
                                                                      y_val_all=y_val_all,
                                                                      X_test_all=X_test_all,
                                                                      y_test_all=y_test_all,
                                                                      chckpt_path=chckpt_path)

            self.acc_patient_folds.append(acc_patient)
            self.precision_patient_folds.append(pre_patient)
            self.recall_patient_folds.append(recall_patient)
            self.fscore_patient_folds.append(fscore_patine)

            for ss in range(3):
                self.accs[fold, ss] = accs_lst[ss]
                self.precision[fold, ss] = pre_lst[ss]
                self.recall[fold, ss] = recall_lst[ss]
                self.fscore[fold, ss] = fscore_lst[ss]

            self.last_epochs[fold, :] = last_epoch_tmp

        self.save_result()

        return self.accs[:, 1].mean()
