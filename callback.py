import constants
from Utils import general_utils

import os, glob, json
import tensorflow as tf
from sklearn.metrics import roc_auc_score

################################################################################
# Return information about the loss and accuracy
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self, root_path, savedModelName, textFolderPath, acc="acc"):
        self.batch_losses = []
        self.batch_acc = []
        self.root_path = root_path
        self.savedModelName = savedModelName
        self.textFolderPath = textFolderPath
        self.folderOfSavedModel = self.savedModelName[:self.savedModelName.rfind("/")+1]
        self.modelName = self.savedModelName[self.savedModelName.rfind("/"):]
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

        tmpSavedModels = glob.glob(self.savedModelName+constants.suffix_partial_weights+"*.h5")
        if len(tmpSavedModels) > 1: # just to be sure and not delete everything
            for file in tmpSavedModels:
                if self.savedModelName+constants.suffix_partial_weights in file:
                    tmpEpoch = general_utils.getEpochFromPartialWeightFilename(file)
                    if tmpEpoch < epoch: # Remove the old saved weights
                        os.remove(file)

################################################################################
#
class RocCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, model, sample_weight, savedModelName, textFolderPath):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.model = model
        self.sample_weight = sample_weight

        self.savedModelName = savedModelName
        self.modelName = self.savedModelName[self.savedModelName.rfind("/"):]

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
def modelCheckpoint(filename, monitor, period):
    return tf.keras.callbacks.ModelCheckpoint(
            filename+constants.suffix_partial_weights+"{epoch:02d}.h5",
            monitor=monitor,
            verbose=constants.getVerbose(),
            save_best_only=True,
            mode='auto',
            period=period
    )

################################################################################
# Stop the training if the "monitor" quantity does NOT change of a "min_delta"
# after a number of "patience" epochs
def earlyStopping(monitor, min_delta, patience):
    return tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=constants.getVerbose(),
            mode="auto"
    )

################################################################################
# Reduce learning rate when a metric has stopped improving.
def reduceLROnPlateau(monitor, factor, patience, min_delta, cooldown, min_lr):
    return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=constants.getVerbose(),
            mode='auto',
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr
    )
