import constants
from Utils import callback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import optimizers
import keract

################################################################################
# Return the optimizer based on the setting
def getOptimizer(optInfo):
    optimizer = None
    if optInfo["name"].lower()=="adam":
        optimizer = optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"],
            beta_2=optInfo["beta_2"],
            epsilon=None if optInfo["epsilon"]=="None" else optInfo["epsilon"],
            decay=optInfo["decay"],
            amsgrad=False
        )
    elif optInfo["name"].lower()=="sgd":
        optimizer = optimizers.SGD(
            learning_rate=optInfo["learning_rate"],
            decay=optInfo["decay"],
            momentum=optInfo["momentum"],
            nesterov=True if optInfo["nesterov"]=="True" else False
        )

    return optimizer

################################################################################
# Return the callbacks defined in the setting
def getCallbacks(info, root_path, filename, textFolderPath, dataset, sample_weights):
    cbs = []
    for key in info.keys():
        # save the weights
        if key=="ModelCheckpoint":
            cbs.append(callback.modelCheckpoint(filename, info[key]["monitor"], info[key]["mode"], info[key]["period"]))
        # stop if the monitor is not improving
        elif key=="EarlyStopping":
            cbs.append(callback.earlyStopping(info[key]["monitor"], info[key]["min_delta"], info[key]["patience"]))
        # reduce the learning rate if the monitor is not improving
        elif key=="ReduceLROnPlateau":
            cbs.append(callback.reduceLROnPlateau(info[key]["monitor"], info[key]["factor"], info[key]["patience"], info[key]["min_delta"], info[key]["cooldown"], info[key]["min_lr"]))
        # reduce learning_rate every fix number of epochs
        elif key=="LearningRateScheduler":
            cbs.append(callback.LearningRateScheduler(info[key]["decay_step"], info[key]["decay_rate"]))
        # collect info
        elif key=="CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, textFolderPath, info[key]["acc"]))
        elif key=="RocCallback":
            training_data = (dataset["train"]["data"], dataset["train"]["labels"])
            validation_data = (dataset["val"]["data"], dataset["val"]["labels"])
            # # TODO: no model passed!
            # # TODO: filename is different (is the TMP_MODELS not MODELS folder)
            cbs.append(callback.RocCallback(training_data, validation_data, model, sample_weights, filename, textFolderPath))

    return cbs

################################################################################
# Fit the model
def fitModel(model, dataset, batch_size, epochs, listOfCallbacks, sample_weights, initial_epoch, save_activation_filter, intermediate_activation_path, use_multiprocessing):
    validation_data = None
    if dataset["val"]["data"] is not None and dataset["val"]["labels"] is not None: validation_data = (dataset["val"]["data"], dataset["val"]["labels"])

    training = model.fit(dataset["train"]["data"],
                dataset["train"]["labels"],
                batch_size=batch_size,
                epochs=epochs,
                callbacks=listOfCallbacks,
                shuffle=True,
                validation_data=validation_data,
                sample_weight=sample_weights,
                initial_epoch=initial_epoch,
                verbose=constants.getVerbose(),
                use_multiprocessing=use_multiprocessing)

    if save_activation_filter: saveActivationFilter(model, shape=tuple(np.array(dataset["train"]["data"][0].shape)), intermediate_activation_path=intermediate_activation_path)

    return training

################################################################################
# Train the model only with a single batch
def trainOnBatch(model, x, y, sample_weights):
    ret = model.train_on_batch(
        x,
        y,
        sample_weight=sample_weights,
        reset_metrics=False
    )

    return model, ret

################################################################################
#
def saveActivationFilter(model, shape, intermediate_activation_path):
    x = np.random.uniform(size=(1,)+shape)
    activations = keract.get_activations(model, x) # call to fetch the activations of the model.
    keract.display_activations(activations, save=True, directory=intermediate_activation_path)

################################################################################
# For plotting the loss and accuracy of the trained model
def plotLossAndAccuracy(nn, p_id):
    for key in nn.train.history.keys():
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(nn.train.history[key],'r',linewidth=3.0)
        ax.legend([key],fontsize=10)
        ax.set_xlabel('Epochs ', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.set_title(key + 'Curves', fontsize=16)

        fig.savefig(nn.savePlotFolder+nn.getNNID(p_id)+"_"+key+"_"+str(constants.SLICING_PIXELS)+"_"+str(constants.getM())+"x"+str(constants.getN())+".png")
        plt.close(fig)

################################################################################
# For plotting the loss and accuracy of the trained model
def plotMetrics(nn, p_id, list_metrics):
    for metric in list_metrics:
        key = metric["name"]
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(metric["val"],'r',linewidth=3.0)
        ax.legend([key],fontsize=10)
        ax.set_xlabel('Batch ', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.set_title(key + 'Curves', fontsize=16)

        fig.savefig(nn.savePlotFolder+nn.getNNID(p_id)+"_"+key+"_"+str(constants.SLICING_PIXELS)+"_"+str(constants.getM())+"x"+str(constants.getN())+".png")
        plt.close(fig)
