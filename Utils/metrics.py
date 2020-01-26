import constants
from Utils import general_utils, callback

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix, multilabel_confusion_matrix

import time

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
# REAL Dice coefficient = (2*|X & Y|)/ (|X|+ |Y|)
# Calculate the real value for the Dice coefficient, but it returns lower values than the other dice_coef + lower specificity and precision
# == to F1 score for boolean values
def dice_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    denom = (K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) + 1)
    return  (2. * intersection + 1) / denom

# TODO:
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

def jaccard_index(tn, fn, fp, tp):
    f = f1(tn, fn, fp, tp)
    return (f+1e-07)/(2-f+1e-07)

def jaccard_index_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return jaccard_index(tn, fn, fp, tp)

def jaccard_index_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    print("jacc core", tn, fn, fp, tp)
    return jaccard_index(tn, fn, fp, tp)

def jaccard_index_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return jaccard_index(tn, fn, fp, tp)

################################################################################
# Function that calculate the metrics for the SENSITIVITY
# ALSO CALLED "RECALL"!
def sensitivity(tn, fn, fp, tp):
    return (tp+1e-07) / (tp+fn+1e-07)

def sensitivity_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return sensitivity(tn, fn, fp, tp)

def sensitivity_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    print("sens core", tn, fn, fp, tp)
    return sensitivity(tn, fn, fp, tp)

def sensitivity_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return sensitivity(tn, fn, fp, tp)


################################################################################
# Function that calculate the metrics for the SPECIFICITY
def specificity(tn, fn, fp, tp):
    return (tn+1e-07) / (tn+fp+1e-07)

def specificity_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return specificity(tn, fn, fp, tp)

def specificity_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    print("spec core", tn, fn, fp, tp)
    return specificity(tn, fn, fp, tp)

def specificity_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return specificity(tn, fn, fp, tp)

################################################################################
# Function that calculate the metrics for the PRECISION
def precision(tn, fn, fp, tp):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    precision = (tp+1e-07)/(tp+fp+1e-07)
    return precision

def precision_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return precision(tn, fn, fp, tp)

def precision_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    print("prec core", tn, fn, fp, tp)
    return precision(tn, fn, fp, tp)

def precision_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return precision(tn, fn, fp, tp)

################################################################################
# Function that calculate the metrics for the F1 SCORE
def f1(tn, fn, fp, tp):
    prec = precision(tn, fn, fp, tp)
    recall = sensitivity(tn, fn, fp, tp)
    return 2*(((prec*recall)+1e-07)/(prec+recall+1e-07))

def f1_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return f1(tn, fn, fp, tp)

def f1_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    print("f1 core", tn, fn, fp, tp)
    return f1(tn, fn, fp, tp)

def f1_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return f1(tn, fn, fp, tp)

################################################################################
# Function that calculate the metrics for the accuracy
def accuracy(tn, fn, fp, tp):
    return (tp+tn+1e-07)/(tn+fn+tp+fn+1e-07)

def accuracy_penumbra(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 2)
    return accuracy(tn, fn, fp, tp)

def accuracy_core(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 3)
    return accuracy(tn, fn, fp, tp)

def accuracy_penumbracore(y_true, y_pred):
    tn, fn, fp, tp = mappingPrediction(y_true, y_pred, 4)
    return accuracy(tn, fn, fp, tp)

################################################################################
# function to convert the prediction and the ground truth in a confusion matrix
def mappingPrediction(y_true, y_pred, label):
    conf_matr = np.zeros(shape=(2,2))
    tn, fp, fn, tp = 0,0,0,0

    or_yt = y_true
    or_yp = y_pred

    if label==2: # penumbra
        y_true, y_pred = thresholdingPenumbra(np.array(y_true), np.array(y_pred))
    elif label==3: # Core
        y_true, y_pred = thresholdingCore(np.array(y_true), np.array(y_pred))
    elif label==4: # Penumbra + Core
        y_true, y_pred = thresholdingPenumbraCore(np.array(y_true), np.array(y_pred))

    for i,_ in enumerate(y_true):
        # tn1, fp1, fn1, tp1 = confusion_matrix(y_true[i], y_pred[i], labels=[0,1]).ravel()
        # tn, fp, fn, tp = tn1+tn, fp1+fp, fn1+fn, tp1+tp

        tmp_conf_matr = multilabel_confusion_matrix(y_true[i], y_pred[i], labels=[0,1])
        # if 76 in or_yt[i] or 150 in or_yt[i]:
        #     print(i)
        #     print(or_yt[i])
        #     print(y_true[i])
        #     print("------")
        #     print(or_yp[i])
        #     print(y_pred[i])
        #     print(tmp_conf_matr)
        #     print(conf_matr)
        #     time.sleep(10)
        conf_matr = tmp_conf_matr[1] + conf_matr

    tn = conf_matr[0][0]
    fn = conf_matr[0][1]
    fp = conf_matr[1][0]
    tp = conf_matr[1][1]

    return tn, fn, fp, tp

################################################################################
# function to map the y_true and y_pred
def thresholdingPenumbra(y_true, y_pred):
    # thresPenumbra = constants.PIXELVALUES[2]/constants.PIXELVALUES[0] # = 0.298
    # thresCore = constants.PIXELVALUES[3]/constants.PIXELVALUES[0] # = 0.588
    # eps = 0.1 # abs((thresCore-thresPenumbra)/2) ~ 0.145
    #
    thresPenumbra = constants.PIXELVALUES[2]
    thresCore = constants.PIXELVALUES[3]
    eps = 36
    y_true = np.array(y_true>=(thresPenumbra-eps), dtype="int32") * np.array(y_true<=(thresPenumbra+eps), dtype="int32")
    y_pred = np.array(y_pred>=(thresPenumbra-eps), dtype="int32") * np.array(y_pred<=(thresPenumbra+eps), dtype="int32")
    return (y_true, y_pred)

def thresholdingCore(y_true, y_pred):
    # thresPenumbra = constants.PIXELVALUES[2]/constants.PIXELVALUES[0] # = 0.298
    # thresCore = constants.PIXELVALUES[3]/constants.PIXELVALUES[0] # = 0.588
    # eps = 0.1 # abs((thresCore-thresPenumbra)/2) ~ 0.145
    #
    thresPenumbra = constants.PIXELVALUES[2]
    thresCore = constants.PIXELVALUES[3]
    eps = 36
    eps_plus = 79 # for upper bounding ~229
    y_true = np.array(y_true>=(thresCore-eps), dtype="int32") * np.array(y_true<=(thresCore+eps_plus), dtype="int32")
    y_pred = np.array(y_pred>=(thresCore-eps), dtype="int32") * np.array(y_pred<=(thresCore+eps_plus), dtype="int32")
    return (y_true, y_pred)

def thresholdingPenumbraCore(y_true, y_pred):
    # thresPenumbra = constants.PIXELVALUES[2]/constants.PIXELVALUES[0] # = 0.298
    # thresCore = constants.PIXELVALUES[3]/constants.PIXELVALUES[0] # = 0.588
    # eps =  0.1 # abs((thresCore-thresPenumbra)/2) ~ 0.145
    #
    thresPenumbra = constants.PIXELVALUES[2]
    thresCore = constants.PIXELVALUES[3]
    eps = 36
    eps_plus = 79 # for upper bounding ~229
    y_true = np.array(y_true>=(thresPenumbra-eps), dtype="int32") * np.array(y_true<=(thresCore+eps_plus), dtype="int32")
    y_pred = np.array(y_pred>=(thresPenumbra-eps), dtype="int32") * np.array(y_pred<=(thresCore+eps_plus), dtype="int32")
    return (y_true, y_pred)
