import constants, callback

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

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
def getCallbacks(info, root_path, filename, textFolderPath):
    cbs = []
    for key in info.keys():
        # save the weights
        if key=="ModelCheckpoint":
            cbs.append(callback.modelCheckpoint(filename, info[key]["monitor"], info[key]["period"]))
        # stop if the monitor is not improving
        elif key=="EarlyStopping":
            cbs.append(callback.earlyStopping(info[key]["monitor"], info[key]["min_delta"], info[key]["patience"]))
        # reduce the learning rate if the monitor is not improving
        elif key=="ReduceLROnPlateau":
            cbs.append(callback.reduceLROnPlateau(info[key]["monitor"], info[key]["factor"], info[key]["patience"], info[key]["min_delta"], info[key]["min_lr"]))
        # collect info
        elif key=="CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, textFolderPath, info[key]["acc"]))

    return cbs

################################################################################
# Fit the model
def fitModel(model, dataset, epochs, listOfCallbacks, class_weights, sample_weights, initial_epoch, use_multiprocessing):

    training = model.fit(dataset["train"]["data"],
                dataset["train"]["labels"],
                epochs=epochs,
                callbacks=listOfCallbacks,
                shuffle=True,
                validation_data=(dataset["val"]["data"], dataset["val"]["labels"]),
                sample_weight=sample_weights,
                initial_epoch=initial_epoch,
                verbose=constants.getVerbose(),
                use_multiprocessing=use_multiprocessing)

    return training

################################################################################
# For plotting the loss and accuracy of the trained model
def plotLossAndAccuracy(nn, p_id):

    for key in nn.train.history.keys():
        #plt.figure(figsize=[8,6])
        plt.plot(nn.train.history[key],'r',linewidth=3.0)
        plt.legend([key],fontsize=10)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel(key,fontsize=16)
        plt.title(key + 'Curves',fontsize=16)

        plt.savefig(nn.savePlotFolder+nn.getNNID(p_id)+"_"+key+"_"+str(constants.SLICING_PIXELS)+"_"+str(constants.M)+"x"+str(constants.N)+".png")

################################################################################
################################################################################
################################################################################
########################### LOSSES & METRICS ###################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
# Funtion that calculates the DICE coefficient. Important when calculates the different of two images
def dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + 1) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + 1)

################################################################################
# Function that calculates the DICE coefficient loss. Util for the LOSS function during the training of the model (for image in input and output)!
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

################################################################################
# Function that calculate the metrics for the SENSITIVITY
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

################################################################################
# Function that calculate the metrics for the SPECIFICITY
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

################################################################################
# Function that calculate AUC-ROC:
# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def aucroc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
