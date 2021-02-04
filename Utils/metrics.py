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
# Implementation of the Focal loss.
# first proposed here: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
def _focal_loss(y_true, y_pred, alpha):
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
# Function that computes the Tanimoto loss
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
# Return precision as a metric
def precision():
    return tf.keras.metrics.Precision()


################################################################################
# Return recall as a metric
def recall():
    return tf.keras.metrics.Recall()