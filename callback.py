import constants
from Utils import general_utils

import os, glob, json
import tensorflow as tf

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
def reduceLROnPlateau(monitor, factor, patience, min_delta, min_lr):
    return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=constants.getVerbose(),
            mode='auto',
            min_delta=min_delta,
            cooldown=0,
            min_lr=min_lr
    )
