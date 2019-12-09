import constants, callback

import matplotlib.pyplot as plt
import tensorflow as tf
# Run the training, save the weight and plots ?

################################################################################
# Return the optimizer based on the setting
def getOptimizer(optInfo):
    optimizer = None
    if optInfo["name"].lower()=="adam":
        optimizer = tf.keras.optimizers.Adam(
            lr=optInfo["lr"],
            beta_1=optInfo["beta_1"],
            beta_2=optInfo["beta_2"],
            epsilon=None if optInfo["epsilon"]=="None" else optInfo["epsilon"],
            decay=optInfo["decay"],
            amsgrad=False)
    # elif optInfo["name"].lower()=="sgd":
    #     ...
    #     TODO!

    return optimizer

################################################################################
# Return the callbacks defined in the setting
def getCallbacks(info, root_path, filename):
    cbs = []
    for key in info.keys():
        if key=="ModelCheckpoint":
            cbs.append(callback.modelCheckpoint(filename, info[key]["monitor"], info[key]["period"]))
        elif key=="EarlyStopping":
            cbs.append(callback.earlyStopping(info[key]["monitor"], info[key]["min_delta"], info[key]["patience"]))
        elif key=="CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, info[key]["acc"]))

    return cbs

################################################################################
# Fit the model
def fitModel(model, dataset, epochs, listOfCallbacks, class_weights, sample_weights, initial_epoch):

    training = model.fit(dataset["train"]["data"],
                dataset["train"]["labels"],
                epochs=epochs,
                callbacks=listOfCallbacks,
                shuffle=True,
                validation_data=(dataset["val"]["data"], dataset["val"]["labels"]),
                sample_weight=sample_weights,
                initial_epoch=initial_epoch,
                verbose=constants.getVerbose())

    return training

################################################################################
# For plotting the loss and accuracy of the trained model
def plotLossAndAccuracy(nn, p_id):
    key = nn.loss["name"]
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(nn.train.history['loss'],'r',linewidth=3.0)
    plt.plot(nn.train.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=10)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    plt.savefig(nn.savePlotFolder+nn.getNNID(p_id)+"_Loss_"+str(constants.SLICING_PIXELS)+"_"+str(constants.M)+"x"+str(constants.N)+".png")

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(nn.train.history[key],'r',linewidth=3.0)
    plt.plot(nn.train.history["val_"+key],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    plt.savefig(nn.savePlotFolder+nn.getNNID(p_id)+"_Acc_"+str(constants.SLICING_PIXELS)+"_"+str(constants.M)+"x"+str(constants.N)+".png")
