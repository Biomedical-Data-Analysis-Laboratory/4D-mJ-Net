import constants, callback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix

CLASSES = []

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
            amsgrad=False
        )
    elif optInfo["name"].lower()=="sgd":
        optimizer = tf.keras.optimizers.SGD(
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
        # collect info
        elif key=="CollectBatchStats":
            cbs.append(callback.CollectBatchStats(root_path, filename, textFolderPath, info[key]["acc"]))
        elif key=="RocCallback":
            training_data = (dataset["train"]["data"], dataset["train"]["labels"])
            validation_data = (dataset["val"]["data"], dataset["val"]["labels"])
            # # TODO: no model passed!
            # # TODO: filename is different (isthe TMP_MODELS not MODELS folder)
            cbs.append(callback.RocCallback(training_data, validation_data, model, sample_weights, filename, textFolderPath))

    return cbs

################################################################################
#
def setClassesToEvaluate(classes):
    CLASSES = classes

################################################################################
# Fit the model
def fitModel(model, dataset, epochs, listOfCallbacks, classesToEvaluate, sample_weights, initial_epoch, use_multiprocessing):
    validation_data = None
    if dataset["val"]["data"] is not None and dataset["val"]["labels"] is not None: validation_data = (dataset["val"]["data"], dataset["val"]["labels"])


    training = model.fit(dataset["train"]["data"],
                dataset["train"]["labels"],
                epochs=epochs,
                callbacks=listOfCallbacks,
                shuffle=True,
                validation_data=validation_data,
                sample_weight=sample_weights,
                initial_epoch=initial_epoch,
                verbose=constants.getVerbose(),
                use_multiprocessing=use_multiprocessing)

    return training

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
################################################################################
################################################################################
########################### LOSSES & METRICS ###################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
# Funtion that calculates the DICE coefficient. Important when calculates the different of two images
def mod_dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + 1) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + 1)

################################################################################
# Function that calculates the DICE coefficient loss. Util for the LOSS function during the training of the model (for image in input and output)!
def mod_dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

################################################################################
# REAL Dice coefficient = (2*|X & Y|)/ (|X|+ |Y|)
# Calculate the real value for the Dice coefficient, but it returns lower values than the other dice_coef + lower specificity and precision
def dice_coef(y_true, y_pred):

    y_true, y_pred =

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    denom = (K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) + 1)
    return  (2. * intersection + 1) / denom

def dice_coef_loss(y_true, y_pred):
    return 1-mod_dice_coef(y_true, y_pred)

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator

    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

# TODO:
def dice_coef_binary(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 2 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))
# TODO:
def dice_coef_binary_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_binary(y_true, y_pred)

################################################################################
# Function to calculate the Jaccard similarity
# The loss has been modified to have a smooth gradient as it converges on zero.
#     This has been shifted so it converges on 0 and is smoothed to avoid exploding
#     or disappearing gradient.
#     Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
#             = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
#
# http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
def jaccard_distance(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    return (intersection + smooth) / (sum_ - intersection + smooth)

################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred, smooth=1):
    return (1 - jaccard_distance(y_true, y_pred, smooth)) * smooth

################################################################################
# Function that calculate the metrics for the SENSITIVITY
# ALSO CALLED "RECALL"!
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
# Function that calculate the metrics for the PRECISION
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

################################################################################
# Function that calculate the metrics for the F1 SCORE
def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recall = sensitivity(y_true, y_pred)
    return 2*((prec*recall)/(prec+recall+K.epsilon()))


################################################################################
# function to convert the prediction and the ground truth in a confusion matrix
def mappingPrediction(y_true, y_pred):
    if not K.learning_phase():
        y_true = np.array(list(map(thresholding, y_true)))
        y_pred = np.array(list(map(thresholding, y_pred)))
        conf_matr = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    return (y_true, y_pred)


################################################################################
# function to map the y_true and y_pred
def thresholding(x):
    if x>0.8: return 0 #constants.LABELS[0]
    elif x<0.2: return 1 #constants.LABELS[1]
    elif x>=0.2 and x<=0.5: 2 #return constants.LABELS[2]
    else: return 3 #constants.LABELS[3]
