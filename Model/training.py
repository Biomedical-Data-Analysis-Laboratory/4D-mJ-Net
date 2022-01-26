import warnings

import cv2, matplotlib, glob

from Model.constants import *
from Utils import callback, general_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Return the optimizer based on the setting
def getOptimizer(optInfo):
    if optInfo["name"].lower() not in ["adam","sgd","rmsprop","adadelta"]: raise Exception("The optimizer is not in the predefined list")
    optimizer = None

    if optInfo["name"].lower() == "adam":
        optimizer = optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"],
            beta_2=optInfo["beta_2"],
            epsilon=None if optInfo["epsilon"] == "None" else optInfo["epsilon"],
            decay=optInfo["decay"],
            amsgrad=True if "amsgrad" in optInfo.keys() and optInfo["amsgrad"] == "True" else False,
            clipvalue=0.5
        )
    elif optInfo["name"].lower() == "sgd":
        optimizer = optimizers.SGD(
            learning_rate=optInfo["learning_rate"],
            momentum=optInfo["momentum"],
            nesterov=True if optInfo["nesterov"] == "True" else False,
            clipvalue=0.5
        )
    elif optInfo["name"].lower() == "rmsprop":
        optimizer = optimizers.RMSprop(
            learning_rate=optInfo["learning_rate"],
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False
        )
    elif optInfo["name"].lower() == "adadelta":
        optimizer = optimizers.Adadelta(
            learning_rate=optInfo["learning_rate"],
            rho=0.95,
            epsilon=1e-07,
            clipvalue=0.5
        )

    return optimizer


################################################################################
# Return the callbacks defined in the setting
def getCallbacks(info, root_path, filename, textFolderPath, dataset, sample_weights, nn_id, add_for_finetuning):
    # add by default the TerminateOnNaN callback
    cbs = [callback.TerminateOnNaN()]  #, callback.SavePrediction()]

    for key in info.keys():
        # save the weights
        if key == "ModelCheckpoint":
            cbs.append(callback.modelCheckpoint(filename, info[key]["monitor"], info[key]["mode"], info[key]["period"]))
        # stop if the monitor is not improving
        elif key == "EarlyStopping":
            cbs.append(callback.earlyStopping(info[key]["monitor"], info[key]["min_delta"], info[key]["patience"]))
        # reduce the learning rate if the monitor is not improving
        elif key == "ReduceLROnPlateau":
            cbs.append(callback.reduceLROnPlateau(info[key]["monitor"], info[key]["factor"], info[key]["patience"],
                                                  info[key]["min_delta"], info[key]["cooldown"], info[key]["min_lr"]))
        # reduce learning_rate every fix number of epochs
        elif key == "LearningRateScheduler":
            cbs.append(callback.LearningRateScheduler(info[key]["decay_step"], info[key]["decay_rate"]))
        # collect info
        elif key == "CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, textFolderPath, info[key]["acc"]))
        # save the epoch results in a csv file
        elif key == "CSVLogger":
            cbs.append(callback.CSVLogger(textFolderPath, nn_id, add_for_finetuning+info[key]["filename"], info[key]["separator"]))
        elif key == "RocCallback":
            training_data = (dataset["train"]["data"], dataset["train"]["labels"])
            validation_data = (dataset["val"]["data"], dataset["val"]["labels"])
            # # TODO: no model passed!
            # # TODO: filename is different (is the TMP_MODELS not MODELS folder)
            cbs.append(callback.RocCallback(training_data, validation_data, model, sample_weights, filename, textFolderPath))
        # elif key=="TensorBoard": cbs.append(callback.TensorBoard(log_dir=textFolderPath, update_freq=info[key]["update_freq"], histogram_freq=info[key]["histogram_freq"]))

    return cbs


################################################################################
# Fit the model
def fitModel(model, dataset, batch_size, epochs, listOfCallbacks, sample_weights, initial_epoch, save_activation_filter,
             intermediate_activation_path, use_multiprocessing):
    validation_data = None
    if dataset["val"]["data"] is not None and dataset["val"]["labels"] is not None:
        validation_data = (dataset["val"]["data"], dataset["val"]["labels"])

    training = model.fit(dataset["train"]["data"],
                         dataset["train"]["labels"],
                         batch_size=batch_size,
                         epochs=epochs,
                         callbacks=listOfCallbacks,
                         shuffle=True,
                         validation_data=validation_data,
                         sample_weight=sample_weights,
                         initial_epoch=initial_epoch,
                         verbose=1,
                         use_multiprocessing=use_multiprocessing)

    return training


################################################################################
# Function that call a fit_generator to load the training dataset on the fly
def fit_generator(model, train_sequence, val_sequence, steps_per_epoch, validation_steps, epochs, listOfCallbacks,
                  initial_epoch, use_multiprocessing):
    multiplier = 16
    # steps_per_epoch is given by the len(train_sequence)*steps_per_epoch_ratio rounded to the nearest integer
    training = model.fit_generator(
        generator=train_sequence,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_sequence,
        validation_steps=validation_steps,
        callbacks=listOfCallbacks,
        initial_epoch=initial_epoch,
        verbose=1,
        max_queue_size=10*multiplier,
        workers=1*multiplier,
        shuffle=True,
        use_multiprocessing=use_multiprocessing)

    return training


################################################################################
# For plotting the loss and accuracy of the trained model
def plotLossAndAccuracy(nn):
    for key in nn.train.history.keys():
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(nn.train.history[key], 'r', linewidth=3.0)
        ax.legend([key], fontsize=10)
        ax.set_xlabel('Epochs ', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.set_title(key + 'Curves', fontsize=16)

        fig.savefig(nn.savePlotFolder + nn.getNNID() + "_" + key + "_" + str(getSLICING_PIXELS()) + "_" + str(getM()) + "x" + str(getN()) + ".png")
        plt.close(fig)
