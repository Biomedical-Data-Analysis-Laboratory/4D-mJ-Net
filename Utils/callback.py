from Model.constants import *
from Utils import general_utils

import os, glob, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score


################################################################################
# Return information about the loss and accuracy
class CollectBatchStats(callbacks.Callback):
    def __init__(self, root_path, savedModelName, textFolderPath, acc="acc"):
        self.batch_losses = []
        self.batch_acc = []
        self.root_path = root_path
        self.savedModelName = savedModelName
        self.textFolderPath = textFolderPath
        # self.folderOfSavedModel = self.savedModelName[:self.savedModelName.rfind("/")+1]
        self.modelName = self.savedModelName[self.savedModelName.rfind(os.path.sep):]
        self.acc = acc

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs[self.acc])

    # when an epoch is finished, it removes the old saved weights (from previous epochs)
    def on_epoch_end(self, epoch, logs=None):
        # save loss and accuracy in files
        textToSave = "epoch: {}".format(str(epoch))
        for k, v in logs.items():
            textToSave += ", {}: {}".format(k, round(v, 6))
        textToSave += '\n'
        with open(self.textFolderPath+self.modelName+"_logs.txt", "a+") as loss_file:
            loss_file.write(textToSave)

        tmpSavedModels = glob.glob(self.savedModelName + suffix_partial_weights + "*.h5")
        if len(tmpSavedModels) > 1:  # just to be sure and not delete everything
            for file in tmpSavedModels:
                if self.savedModelName+ suffix_partial_weights in file:
                    tmpEpoch = general_utils.getEpochFromPartialWeightFilename(file)
                    if tmpEpoch < epoch: os.remove(file)  # Remove the old saved weights


################################################################################
# Callback for the AUC_ROC
# TODO: NOT working!! <-- use it just for testing
class RocCallback(callbacks.Callback):
    def __init__(self, training_data, validation_data, model, sample_weight, savedModelName, textFolderPath):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.model = model
        self.sample_weight = sample_weight

        self.savedModelName = savedModelName
        self.modelName = self.savedModelName[self.savedModelName.rfind(os.path.sep):]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # TODO: predict_proba does NOT exist!
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train, sample_weight=self.sample_weight)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val, sample_weight=self.sample_weight)

        print('\r roc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')

        with open(self.textFolderPath+self.modelName+"_aucroc.txt", "a+") as aucroc_file:
            aucroc_file.write('\r roc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


################################################################################
# Save the best model every "period" number of epochs
def modelCheckpoint(filename, monitor, mode, period):
    return callbacks.ModelCheckpoint(
        filename + suffix_partial_weights + "{epoch:02d}.h5",
            monitor=monitor,
            verbose=getVerbose(),
            save_best_only=True,
            mode=mode,
            period=period
    )


################################################################################
# Stop the training if the "monitor" quantity does NOT change of a "min_delta"
# after a number of "patience" epochs
def earlyStopping(monitor, min_delta, patience):
    return callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=getVerbose(),
            mode="auto"
    )


################################################################################
# Reduce learning rate when a metric has stopped improving.
def reduceLROnPlateau(monitor, factor, patience, min_delta, cooldown, min_lr):
    return callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=getVerbose(),
            mode='auto',
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr
    )


################################################################################
# Reduce the learning rate every decay_step of a certain decay_rate
def LearningRateScheduler(decay_step, decay_rate):
    def lr_scheduler(epoch, lr):
        if epoch % decay_step == 0 and epoch: return lr * decay_rate
        return lr
    return callbacks.LearningRateScheduler(lr_scheduler, verbose=getVerbose())


################################################################################
# If you have installed TensorFlow with pip, you should be able to launch TensorBoard from the command line:
# tensorboard --logdir="/home/stud/lucat/PhD_Project/Stroke_segmentation/SAVE/"
def TensorBoard(log_dir, histogram_freq, update_freq):
    return callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_freq,
        update_freq=update_freq
    )


################################################################################
# Callback that terminates training when a NaN loss is encountered.
def TerminateOnNaN():
    return callbacks.TerminateOnNaN()


################################################################################
# Callback that streams epoch results to a CSV file.
def CSVLogger(textFolderPath, nn_id, filename, separator):
    return callbacks.CSVLogger(textFolderPath+nn_id+general_utils.getSuffix()+filename, separator=separator, append=True)


################################################################################
#
class SavePrediction(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._get_pred = None
        self.preds = dict()
        self.n_pxl = 0

    def _pred_callback(self, preds):
        batch = K.eval((K.argmax(preds)*255)/(getN_CLASSES()-1))
        for val_idx in range(batch.shape[0]):
            vals = dict(zip(*np.unique(batch[val_idx, :, :], return_counts=True)))
            for key in vals.keys():
                if key not in self.preds.keys(): self.preds[key] = 0
                self.preds[key] += vals[key]
                self.n_pxl += vals[key]

    def set_model(self, model):
        super().set_model(model)
        if self._get_pred is None: self._get_pred = self.model.outputs[0]
        # if self._get_inp is None: self._get_inp = self.model.inputs[0]
        # if self._get_target is None: self._get_target = self.model._targets[0]

    def on_test_begin(self, logs):
        # pylint: disable=protected-access
        self.model._make_test_function()
        # pylint: enable=protected-access
        if self._get_pred not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self._get_pred)
            self.model.test_function.fetch_callbacks[self._get_pred] = self._pred_callback

    def on_test_end(self, logs):
        if self._get_pred in self.model.test_function.fetches: self.model.test_function.fetches.remove(self._get_pred)
        if self._get_pred in self.model.test_function.fetch_callbacks: self.model.test_function.fetch_callbacks.pop(self._get_pred)

        print("\n {0} - {1}".format(self.preds, self.n_pxl/(getIMAGE_HEIGHT()*getIMAGE_WIDTH())))
        self.preds = dict()
        self.n_pxl = 0

