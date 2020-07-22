import constants
from Utils import general_utils, callback

import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, utils
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, auc, multilabel_confusion_matrix


################################################################################
# Funtion that calculates the DICE coefficient. Important when calculates the different of two images
def mod_dice_coef(y_true, y_pred, epsilon=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    # axes = tuple(range(1, len(y_pred.shape)-1))
    # numerator = 2. * K.sum(K.abs(y_pred * y_true), axis=axes)
    # denominator = (K.sum(K.square(y_pred) + K.square(y_true), axis=axes) + epsilon)
    # return (numerator/denominator)

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    denom = (K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + 1)

    return (2. * intersection + 1) / denom

################################################################################
# REAL Dice coefficient = (2*|X & Y|)/ (|X|+ |Y|)
# Calculate the real value for the Dice coefficient, but it returns lower values than the other dice_coef + lower specificity and precision
# == to F1 score for boolean values
def dice_coef(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1))
    intersection = 2. * K.sum(K.abs(y_true * y_pred), axis=axes)
    denom = (K.sum(K.abs(y_true) + K.abs(y_pred), axis=axes) + epsilon)
    return  (intersection/denom)

################################################################################
# Implementation of the Tversky Index (TI),
# which is a asymmetric similarity measure that is a generalisation of the dice coefficient and the Jaccard index.
# Function taken and modified from here: https://github.com/robinvvinod/unet/
def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    beta = 1-alpha
    true_pos = K.sum(y_true * y_pred, axis=-1)
    false_neg = K.sum(y_true * (1 - y_pred), axis=-1)
    false_pos = K.sum((1 - y_true) * y_pred, axis=-1)

    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

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

################################################################################
# Function that calculate the metrics for the CATEGORICAL CROSS ENTROPY
def categorical_crossentropy(y_true, y_pred):
    return metrics.categorical_accuracy(y_true, y_pred)

################################################################################
# Function that calculate the metrics for the WEIGHTED CATEGORICAL CROSS ENTROPY
def weighted_categorical_cross_entropy(y_true, y_pred):
    lambda_0 = 1
    lambda_1 = 1e-6
    lambda_2 = 1e-5
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS)

    cce = categorical_crossentropy(y_true, y_pred)
    weights = K.cast(tf.reduce_sum(class_weights*y_true),'float32')
    wcce = (weights * cce)/weights
    l1_norm =  K.sum(K.abs(y_true - y_pred))
    l2_norm =  K.sum(K.square(y_true - y_pred))

    return ((lambda_0*wcce) + (lambda_1*l1_norm) + (lambda_2*l2_norm))

################################################################################
# Function that calculate the metrics for the SENSITIVITY
# ALSO CALLED "RECALL"!
def sensitivity(tn, fn, fp, tp):
    return (tp+1e-07) / (tp+fn+1e-07)

################################################################################
# Function that calculate the metrics for the SPECIFICITY
def specificity(tn, fn, fp, tp):
    return (tn+1e-07) / (tn+fp+1e-07)

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

################################################################################
# Function that calculate the metrics for the F1 SCORE
def f1(tn, fn, fp, tp):
    prec = precision(tn, fn, fp, tp)
    recall = sensitivity(tn, fn, fp, tp)
    return 2*(((prec*recall)+1e-07)/(prec+recall+1e-07))

################################################################################
# Function that calculate the metrics for the accuracy
def accuracy(tn, fn, fp, tp):
    return (tp+tn+1e-07)/(tn+fn+tp+fn+1e-07)

################################################################################
# Function that calculate the metrics for the average precision
def mAP(y_true, y_pred, use_background_in_statistics, label):
    # if label==2: # penumbra
    #     y_true, y_pred = thresholdingPenumbra(np.array(y_true), np.array(y_pred))
    # elif label==3: # Core
    #     y_true, y_pred = thresholdingCore(np.array(y_true), np.array(y_pred))
    # elif label==4: # Penumbra + Core
    #     y_true, y_pred = thresholdingPenumbraCore(np.array(y_true), np.array(y_pred))
    if label==4: label=None
    y_true, y_pred = thresholding(np.array(y_true), np.array(y_pred), use_background_in_statistics, label)

    if label==None:
        y_true_p = np.array(y_true==2, dtype="int32")
        y_true_c = np.array(y_true==3, dtype="int32")
        y_true = y_true_p+y_true_c
        y_pred_p = np.array(y_pred==2, dtype="int32")
        y_pred_c = np.array(y_pred==3, dtype="int32")
        y_pred = y_pred_p+y_pred_c


    return average_precision_score(y_true, y_pred)

# def AUC(y_true, y_pred, label):
#     if label==2: # penumbra
#         y_true, y_pred = thresholdingPenumbra(np.array(y_true), np.array(y_pred))
#     elif label==3: # Core
#         y_true, y_pred = thresholdingCore(np.array(y_true), np.array(y_pred))
#     elif label==4: # Penumbra + Core
#         y_true, y_pred = thresholdingPenumbraCore(np.array(y_true), np.array(y_pred))
#
#     return auc(y_true, y_pred)

def ROC_AUC(y_true, y_pred, use_background_in_statistics, label):
    # if label==2: # penumbra
    #     y_true, y_pred = thresholdingPenumbra(np.array(y_true), np.array(y_pred))
    # elif label==3: # Core
    #     y_true, y_pred = thresholdingCore(np.array(y_true), np.array(y_pred))
    # elif label==4: # Penumbra + Core
    #     y_true, y_pred = thresholdingPenumbraCore(np.array(y_true), np.array(y_pred))
    if label==4: label=None
    y_true, y_pred = thresholding(np.array(y_true), np.array(y_pred), use_background_in_statistics, label)

    if label==None:
        y_true_p = np.array(y_true==2, dtype="int32")
        y_true_c = np.array(y_true==3, dtype="int32")
        y_true = y_true_p+y_true_c
        y_pred_p = np.array(y_pred==2, dtype="int32")
        y_pred_c = np.array(y_pred==3, dtype="int32")
        y_pred = y_pred_p+y_pred_c

    try:
        roc_score = roc_auc_score(y_true, y_pred)
        return roc_score
    except:
        return 0

################################################################################
# function to convert the prediction and the ground truth in a confusion matrix
def mappingPrediction(y_true, y_pred, use_background_in_statistics, epsilons, percEps, label):
    conf_matr = np.zeros(shape=(2,2))
    tn, fp, fn, tp = 0,0,0,0

    y_true, y_pred = thresholding(np.array(y_true), np.array(y_pred), use_background_in_statistics, epsilons, percEps, label)

    tmp_conf_matr = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    if label!=4: conf_matr = tmp_conf_matr[label] + conf_matr
    else: conf_matr = tmp_conf_matr[2] + tmp_conf_matr[3] + conf_matr

    tn = conf_matr[0][0]
    fn = conf_matr[1][0]
    fp = conf_matr[0][1]
    tp = conf_matr[1][1]

    return tn, fn, fp, tp

################################################################################
# function to map the y_true and y_pred
def thresholding(y_true, y_pred, use_background_in_statistics, epsilons, percEps, label=None):
    thresBack = constants.PIXELVALUES[0]
    thresBrain = constants.PIXELVALUES[1]-1
    thresPenumbra = constants.PIXELVALUES[2]
    thresCore = constants.PIXELVALUES[3]
    eps1, eps2, eps3 = epsilons[0]

    y_true_brain = np.zeros_like(y_true)
    y_pred_brain = np.zeros_like(y_pred)
    y_true_p = np.zeros_like(y_true)
    y_pred_p = np.zeros_like(y_pred)
    y_true_c = np.zeros_like(y_true)
    y_pred_c = np.zeros_like(y_pred)

    y_true_brain = np.array(y_true<=(thresBrain+eps1), dtype="int32")
    y_pred_brain = np.array(y_pred<=(thresBrain+eps1), dtype="int32")

    # if label==None it means that the y_true & y_pred are coming from label=4 in ROC and mAP ...
    if percEps==None: # no need to calculate the ROC with thresholding
        y_true_p = np.array(y_true>(thresBrain+eps1), dtype="int32") * np.array(y_true<=(thresPenumbra+eps2), dtype="int32")
        y_pred_p = np.array(y_pred>(thresBrain+eps1), dtype="int32") * np.array(y_pred<=(thresPenumbra+eps2), dtype="int32")
    else:
        if label==2: # penumbra 
            upperBound = ((thresBack-thresPenumbra)*percEps)/100
            lowerBound = ((thresPenumbra-thresBrain)*percEps)/100
            y_true_brain = np.array(y_true<=(thresPenumbra-lowerBound), dtype="int32")
            y_pred_brain = np.array(y_pred<=(thresPenumbra-lowerBound), dtype="int32")

            y_true_p = np.array(y_true>(thresPenumbra-lowerBound), dtype="int32") * np.array(y_true<=(thresPenumbra+upperBound), dtype="int32")
            y_pred_p = np.array(y_pred>(thresPenumbra-lowerBound), dtype="int32") * np.array(y_pred<=(thresPenumbra+upperBound), dtype="int32")

            y_true_c = np.array(y_true>(thresPenumbra+upperBound), dtype="int32")
            y_pred_c = np.array(y_pred>(thresPenumbra+upperBound), dtype="int32")

    if percEps==None: # no need to calculate the ROC with thresholding
        y_true_c = np.array(y_true>(thresPenumbra+eps2), dtype="int32") * np.array(y_true<=(thresCore+eps3), dtype="int32")
        y_pred_c = np.array(y_pred>(thresPenumbra+eps2), dtype="int32") * np.array(y_pred<=(thresCore+eps3), dtype="int32")
    else:
        if label==3: # core
            upperBound = ((thresBack-thresCore)*percEps)/100
            lowerBound = ((thresCore-thresBrain)*percEps)/100
            y_true_brain = np.array(y_true<=(thresCore-lowerBound), dtype="int32")
            y_pred_brain = np.array(y_pred<=(thresCore-lowerBound), dtype="int32")

            y_true_c = np.array(y_true>(thresCore-lowerBound), dtype="int32") * np.array(y_pred<=(thresCore+upperBound), dtype="int32")
            y_pred_c = np.array(y_pred>(thresCore-lowerBound), dtype="int32") * np.array(y_pred<=(thresCore+upperBound), dtype="int32")

    y_true_p = y_true_p * 2
    y_pred_p = y_pred_p * 2
    y_true_c = y_true_c * 3
    y_pred_c = y_pred_c * 3

    y_true = y_true_brain+y_true_p+y_true_c
    y_pred = y_pred_brain+y_pred_p+y_pred_c

    y_true_new = np.empty([0])
    y_pred_new = np.empty_like(y_true_new)

    # REMOVE BACKGROUND FROM y_true & y_pred
    for row in range(0,y_true.shape[0]-1):
        index_back = np.where(y_true[row]==0)[0]

        if not use_background_in_statistics:
            y_true_new = np.append(y_true_new, np.delete(y_true[row], index_back))
            y_pred_new = np.append(y_pred_new, np.delete(y_pred[row], index_back))
        else:
            y_true_new = np.append(y_true_new, y_true[row])
            y_pred_new = np.append(y_pred_new, y_pred[row])

    return (y_true_new, y_pred_new)

################################################################################
# function to map the y_true and y_pred for penumbra
# def thresholdingPenumbra(y_true, y_pred):
#     thresPenumbra = constants.PIXELVALUES[2]
#     thresCore = constants.PIXELVALUES[3]
#     eps = 36
#
#     y_true_brain = np.array(y_true<=(thresBrain+eps), dtype="int32")
#     y_pred_brain = np.array(y_pred<=(thresBrain+eps), dtype="int32")
#     y_true_p = np.array(y_true>=(thresPenumbra-eps), dtype="int32") * np.array(y_true<=(thresPenumbra+eps), dtype="int32")
#     y_pred_p = np.array(y_pred>=(thresPenumbra-eps), dtype="int32") * np.array(y_pred<=(thresPenumbra+eps), dtype="int32")
#
#     y_true_p = y_true_p * 2
#     y_pred_p = y_pred_p * 2
#
#
#     return (y_true, y_pred)
#
# ################################################################################
# # function to map the y_true and y_pred for core
# def thresholdingCore(y_true, y_pred):
#     thresPenumbra = constants.PIXELVALUES[2]
#     thresCore = constants.PIXELVALUES[3]
#     eps = 36
#     eps_plus = 79 # for upper bounding ~229
#     y_true = np.array(y_true>=(thresCore-eps), dtype="int32") * np.array(y_true<=(thresCore+eps_plus), dtype="int32")
#     y_pred = np.array(y_pred>=(thresCore-eps), dtype="int32") * np.array(y_pred<=(thresCore+eps_plus), dtype="int32")
#     return (y_true, y_pred)
#
# ################################################################################
# # function to map the y_true and y_pred for penumbra & core
# def thresholdingPenumbraCore(y_true, y_pred):
#     thresPenumbra = constants.PIXELVALUES[2]
#     thresCore = constants.PIXELVALUES[3]
#     eps = 36
#     eps_plus = 79 # for upper bounding ~229
#     y_true = np.array(y_true>=(thresPenumbra-eps), dtype="int32") * np.array(y_true<=(thresCore+eps_plus), dtype="int32")
#     y_pred = np.array(y_pred>=(thresPenumbra-eps), dtype="int32") * np.array(y_pred<=(thresCore+eps_plus), dtype="int32")
#     return (y_true, y_pred)
