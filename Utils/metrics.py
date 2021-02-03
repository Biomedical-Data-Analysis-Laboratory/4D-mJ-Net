import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import constants

import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, utils
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, auc, multilabel_confusion_matrix


################################################################################
# Function that calculates the DICE coefficient. Important when calculates the different of two images
def _squared_dice_coef(y_true, y_pred, class_weights):
    """
    Compute weighted squared Dice loss.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    axis_to_reduce = list(range(1, K.ndim(y_pred)))  # Reduce all axis but first (batch)
    numerator = y_true * y_pred * class_weights  # Broadcasting
    numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

    denominator = (K.square(y_true) + K.square(y_pred)) * class_weights  # Broadcasting
    denominator = K.sum(denominator, axis=axis_to_reduce)
    return numerator / denominator


def squared_dice_coef(y_true, y_pred):
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS, dtype=tf.float32)
    return _squared_dice_coef(y_true, y_pred, class_weights)


def sdc_rest(y_true, y_pred):
    class_weights = tf.constant([[1,1,0,0]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[1, 0, 0]], dtype=tf.float32)
    return _squared_dice_coef(y_true, y_pred, class_weights)


def sdc_p(y_true, y_pred):
    class_weights = tf.constant([[0,0,1,0]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[0, 1, 0]], dtype=tf.float32)
    return _squared_dice_coef(y_true, y_pred, class_weights)


def sdc_c(y_true, y_pred):
    class_weights = tf.constant([[0, 0, 0, 1]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[0, 0, 1]], dtype=tf.float32)
    return _squared_dice_coef(y_true, y_pred, class_weights)


################################################################################
# REAL Dice coefficient = (2*|X & Y|)/ (|X|+ |Y|)
# Calculate the real value for the Dice coefficient,
# but it returns lower values than the other dice_coef + lower specificity and precision
# == to F1 score for boolean values
def dice_coef(y_true, y_pred):
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS, dtype=tf.float32)
    """ Compute weighted Dice loss. """

    axis_to_reduce = list(range(1, K.ndim(y_pred)))  # Reduce all axis but first (batch)
    numerator = y_true * y_pred * class_weights  # Broadcasting
    numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

    denominator = (y_true + y_pred) * class_weights  # Broadcasting
    denominator = K.sum(denominator, axis=axis_to_reduce)
    return numerator / denominator


################################################################################
# Implementation of the Tversky Index (TI),
# which is a asymmetric similarity measure that is a generalisation of the dice coefficient and the Jaccard index.
# Function taken and modified from here: https://github.com/robinvvinod/unet/
def _tversky_coef(y_true, y_pred, class_weights, smooth=1.):
    alpha = constants.focal_tversky_loss["alpha"]
    beta = 1-alpha

    axis_to_reduce = list(range(1, K.ndim(y_pred)))  # All axis but first (batch)
    numerator = (y_true * y_pred) * class_weights  # Broadcasting
    numerator = K.sum(numerator, axis=axis_to_reduce)
    denominator = (y_true * y_pred) + alpha * (y_true * (1 - y_pred)) + beta * ((1 - y_true) * y_pred)
    denominator *= class_weights  # Broadcasting
    denominator = K.sum(denominator, axis=axis_to_reduce)

    return (numerator + smooth) / (denominator + smooth)


def tversky_coef(y_true, y_pred):
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS, dtype=tf.float32)
    return _tversky_coef(y_true, y_pred, class_weights)


def tversky_rest(y_true, y_pred):
    class_weights = tf.constant([[1,1,0,0]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[1, 0, 0]], dtype=tf.float32)
    return _tversky_coef(y_true, y_pred, class_weights)


def tversky_p(y_true, y_pred):
    class_weights = tf.constant([[0,0,1,0]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[0, 1, 0]], dtype=tf.float32)
    return _tversky_coef(y_true, y_pred, class_weights)


def tversky_c(y_true, y_pred):
    class_weights = tf.constant([[0, 0, 0, 1]], dtype=tf.float32)
    if constants.N_CLASSES == 3: class_weights = tf.constant([[0, 0, 1]], dtype=tf.float32)
    return _tversky_coef(y_true, y_pred, class_weights)


################################################################################
# Function to calculate the Jaccard similarity
# The loss has been modified to have a smooth gradient as it converges on zero.
#     This has been shifted so it converges on 0 and is smoothed to avoid exploding
#     or disappearing gradient.
#     Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
#             = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
#
# http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jaccard_index(tn, fn, fp, tp):
    f = f1(tn, fn, fp, tp)
    return f/(2-f+1e-07)


################################################################################
# Function that calculate the metrics for the CATEGORICAL CROSS ENTROPY
def categorical_crossentropy(y_true, y_pred):
    return metrics.categorical_accuracy(y_true, y_pred)


################################################################################
# Function that calculate the metrics for the WEIGHTED CATEGORICAL CROSS ENTROPY
def weighted_categorical_cross_entropy(y_true, y_pred, epsilon=1e-7):
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS, dtype=tf.float32)
    lambda_0 = 1
    lambda_1 = 1e-6
    lambda_2 = 1e-5

    cce = categorical_crossentropy(y_true, y_pred)
    weights = K.cast(tf.reduce_sum(class_weights*y_true),'float32')+epsilon
    wcce = (weights * cce)/weights
    l1_norm = K.sum(K.abs(y_true - y_pred))+epsilon
    l2_norm = K.sum(K.square(y_true - y_pred))+epsilon

    return (lambda_0 * wcce) + (lambda_1 * l1_norm) + (lambda_2 * l2_norm)


################################################################################
#
def _focal_loss(y_true, y_pred, alpha, epsilon=1e-6):
    """ Compute focal loss. """
    gamma = tf.constant(constants.GAMMA,dtype=y_pred.dtype)
    axis_to_reduce = list(range(1, K.ndim(y_pred)))
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -(y_true * K.log(y_pred))
    f_loss = alpha * K.pow((1 - y_pred), gamma) * cross_entropy
    # Average over each data point/image in batch
    f_loss = K.mean(f_loss, axis=axis_to_reduce)

    return f_loss


def focal_loss(y_true, y_pred):
    alpha = tf.constant(constants.ALPHA, dtype=y_pred.dtype)
    return _focal_loss(y_true, y_pred, alpha)


def focal_rest(y_true, y_pred):
    alpha = tf.constant(constants.ALPHA, dtype=y_pred.dtype)
    if constants.N_CLASSES == 3: alpha = tf.constant([[0.25, 0, 0]], dtype=tf.float32)
    return _focal_loss(y_true, y_pred, alpha)


def focal_p(y_true, y_pred):
    alpha = tf.constant(constants.ALPHA, dtype=y_pred.dtype)
    if constants.N_CLASSES == 3: alpha = tf.constant([[0, 0.25, 0]], dtype=tf.float32)
    return _focal_loss(y_true, y_pred, alpha)


def focal_c(y_true, y_pred):
    alpha = tf.constant(constants.ALPHA, dtype=y_pred.dtype)
    if constants.N_CLASSES == 3: alpha = tf.constant([[0, 0, 0.25]], dtype=tf.float32)
    return _focal_loss(y_true, y_pred, alpha)


################################################################################
#
def tanimoto(y_true, y_pred):
    class_weights = tf.constant(constants.HOT_ONE_WEIGHTS, dtype=tf.float32)
    """
    Compute weighted Tanimoto loss.
    Defined in the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data",
    under 3.2.4. Generalization to multiclass imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf
    """

    axis_to_reduce = list(range(1, K.ndim(y_pred)))  # All axis but first (batch)
    numerator = y_true * y_pred * class_weights
    numerator = K.sum(numerator, axis=axis_to_reduce)

    denominator = (K.square(y_true) + K.square(y_pred) - y_true * y_pred) * class_weights
    denominator = K.sum(denominator, axis=axis_to_reduce)
    return numerator / denominator


################################################################################
# Function that calculate the metrics for the SENSITIVITY
# ALSO CALLED "RECALL"!
def sensitivity(tn, fn, fp, tp):
    return tp/(tp+fn+1e-07)


################################################################################
# Function that calculate the metrics for the SPECIFICITY
def specificity(tn, fn, fp, tp):
    return tn/(tn+fp+1e-07)


################################################################################
# Function that calculate the metrics for the PRECISION
def precision(tn, fn, fp, tp):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    return tp/(tp+fp+1e-07)


################################################################################
# Function that calculate the metrics for the F1 SCORE
def f1(tn, fn, fp, tp):
    prec = precision(tn, fn, fp, tp)
    recall = sensitivity(tn, fn, fp, tp)
    return 2*((prec*recall)/(prec+recall+1e-07))


################################################################################
# Function that calculate the metrics for the accuracy
def accuracy(tn, fn, fp, tp):
    return (tp+tn)/(tn+fn+tp+fn+1e-07)


################################################################################
# Function that calculate the metrics for the average precision
def mAP(y_true, y_pred, use_background_in_statistics, label):
    if label==4: label=None
    y_true, y_pred = thresholding(np.array(y_true), np.array(y_pred), use_background_in_statistics, label)

    if label is None:
        y_true_p = np.array(y_true==2, dtype="int32")
        y_true_c = np.array(y_true==3, dtype="int32")
        y_true = y_true_p+y_true_c
        y_pred_p = np.array(y_pred==2, dtype="int32")
        y_pred_c = np.array(y_pred==3, dtype="int32")
        y_pred = y_pred_p+y_pred_c

    return average_precision_score(y_true, y_pred)


def ROC_AUC(y_true, y_pred, use_background_in_statistics, label):
    if label==4: label=None
    y_true, y_pred = thresholding(np.array(y_true), np.array(y_pred), use_background_in_statistics, label)

    if label is None:
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
    if percEps is None:  # no need to calculate the ROC with thresholding
        y_true_p = np.array(y_true>(thresBrain+eps1), dtype="int32") * np.array(y_true<=(thresPenumbra+eps2), dtype="int32")
        y_pred_p = np.array(y_pred>(thresBrain+eps1), dtype="int32") * np.array(y_pred<=(thresPenumbra+eps2), dtype="int32")
    else:
        if label==2:  # penumbra
            upperBound = ((thresBack-thresPenumbra)*percEps)/100
            lowerBound = ((thresPenumbra-thresBrain)*percEps)/100
            y_true_brain = np.array(y_true<=(thresPenumbra-lowerBound), dtype="int32")
            y_pred_brain = np.array(y_pred<=(thresPenumbra-lowerBound), dtype="int32")

            y_true_p = np.array(y_true>(thresPenumbra-lowerBound), dtype="int32") * np.array(y_true<=(thresPenumbra+upperBound), dtype="int32")
            y_pred_p = np.array(y_pred>(thresPenumbra-lowerBound), dtype="int32") * np.array(y_pred<=(thresPenumbra+upperBound), dtype="int32")

            y_true_c = np.array(y_true>(thresPenumbra+upperBound), dtype="int32")
            y_pred_c = np.array(y_pred>(thresPenumbra+upperBound), dtype="int32")

    if percEps is None:  # no need to calculate the ROC with thresholding
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

    return y_true_new, y_pred_new
