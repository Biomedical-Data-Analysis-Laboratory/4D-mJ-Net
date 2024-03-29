import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Model.constants import *
from Utils import general_utils
import tensorflow as tf
from tensorflow.keras import metrics, utils, losses
import tensorflow.keras.backend as K


################################################################################
# Function that calculates the SOFT DICE coefficient. Important when calculates the different of two images
def _squared_dice_coef(y_true, y_pred, class_weights, is_loss, smooth=100, class_axis=None):
    """
    Compute weighted squared Dice loss.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    if not is_loss: smooth = K.epsilon()
    y_true, y_pred, axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis, is_loss)
    out = 0
    if is_loss:
        for a in axis_to_reduce:
            numerator = 2. * K.sum(K.abs(y_true[:,a,...] * y_pred[:,a,...]))
            denominator = K.sum(K.square(y_true[:,a,...]))+ K.sum(K.square(y_pred[:,a,...]))
            out += ((numerator+smooth) / (denominator+smooth))
    else:
        numerator = 2. * K.sum(K.abs(y_true * y_pred), axis=axis_to_reduce)
        denominator = K.sum(K.square(y_true),axis=axis_to_reduce) + K.sum(K.square(y_pred),axis=axis_to_reduce)
        out = (numerator+smooth) / (denominator+smooth)
    return out


def squared_dice_coef(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights(is_loss=is_loss)
    return _squared_dice_coef(y_true, y_pred, class_weights, is_loss)


def sdc_rest(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _squared_dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=0)


def sdc_p(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _squared_dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=1)


def sdc_c(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _squared_dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=get_n_classes()-1)


################################################################################
# Dice coefficient = (2*|X & Y|)/ (|X|+ |Y|)
# Calculate the real value for the Dice coefficient,
# but it returns lower values than the other dice_coef + lower specificity and precision
# == to F1 score for boolean values
def _dice_coef(y_true, y_pred, class_weights, is_loss, smooth=100, class_axis=None):
    """ Compute weighted Dice loss. """
    if not is_loss: smooth=K.epsilon()
    y_true,y_pred,axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis, is_loss)
    out = 0
    if is_loss:
        for a in axis_to_reduce:
            numerator = 2. * K.sum(K.abs(y_true[:,a,...] * y_pred[:,a,...]))
            denominator = K.sum(y_true[:,a,...] + y_pred[:,a,...])
            out += ((numerator+smooth) / (denominator+smooth))
    else:
        numerator = 2. * K.sum(K.abs(y_true * y_pred), axis=axis_to_reduce)
        denominator = K.sum(y_true + y_pred,  axis=axis_to_reduce)
        out = (numerator + smooth) / (denominator + smooth)
    return out


def dice_coef(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights(is_loss=is_loss)
    return _dice_coef(y_true, y_pred, class_weights, is_loss)


def dc_rest(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=0)


def dc_p(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=1)


def dc_c(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _dice_coef(y_true, y_pred, class_weights, is_loss, class_axis=get_n_classes()-1)


################################################################################
def _tversky_coef_hybrid(y_true, y_pred, class_weights, is_loss, class_axis=None):
    alpha = get_Focal_Tversky()["alpha"]
    beta = 1-alpha
    y_true, y_pred, axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis, is_loss)

    cce_f = categorical_crossentropy(y_true, y_pred)

    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)
    denominator = (y_true * y_pred) + alpha * (y_true * (1 - y_pred)) + beta * ((1 - y_true) * y_pred)
    denominator = K.sum(denominator, axis=axis_to_reduce)

    return cce_f + (numerator/(denominator + K.epsilon()))


def tversky_coef_hybrid(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights(is_loss=is_loss)
    return _tversky_coef_hybrid(y_true, y_pred, class_weights, is_loss)


################################################################################
# Implementation of the Tversky Index (TI),
# which is an asymmetric similarity measure that is a generalisation of the dice coefficient and the Jaccard index.
def _tversky_coef(y_true, y_pred, class_weights, is_loss, class_axis=None):
    alpha = get_Focal_Tversky()["alpha"]
    beta = 1-alpha
    y_true, y_pred, axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis, is_loss)

    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)
    denominator = (y_true * y_pred) + alpha * (y_true * (1 - y_pred)) + beta * ((1 - y_true) * y_pred)
    denominator = K.sum(denominator, axis=axis_to_reduce)

    out = (numerator+ K.epsilon()) / (denominator + K.epsilon())
    return out


def tversky_coef(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights(is_loss=is_loss)
    return _tversky_coef(y_true, y_pred, class_weights, is_loss)


def tversky_rest(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _tversky_coef(y_true, y_pred, class_weights, is_loss, class_axis=0)


def tversky_p(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _tversky_coef(y_true, y_pred, class_weights, is_loss, class_axis=1)


def tversky_c(y_true, y_pred, is_loss=False):
    class_weights = general_utils.get_class_weights()
    return _tversky_coef(y_true, y_pred, class_weights, is_loss, class_axis=get_n_classes()-1)


################################################################################
# Function to calculate the Jaccard similarity
# The loss has been modified to have a smooth gradient as it converges on zero.
#     This has been shifted so it converges on 0 and is smoothed to avoid exploding
#     or disappearing gradient.
#     Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
#             = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
#
# http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
def jaccard_distance(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    return intersection / (sum_ - intersection + K.epsilon())


################################################################################
# Function that calculate the metrics for the CATEGORICAL CROSS ENTROPY
def categorical_crossentropy(y_true, y_pred):
    y_true = K.reshape(y_true, K.stack([K.prod(K.shape(y_true)[:-1]),-1]))
    y_pred = K.reshape(y_pred, K.stack([K.prod(K.shape(y_pred)[:-1]),-1]))
    cce_f = losses.CategoricalCrossentropy()
    return cce_f(y_true, y_pred)


################################################################################
# Function that calculate the metrics for the WEIGHTED CATEGORICAL CROSS ENTROPY
def weighted_categorical_cross_entropy(y_true, y_pred):
    class_weights = tf.constant(1, dtype=K.floatx())
    lambda_0 = 1
    lambda_1 = 1e-6
    lambda_2 = 1e-5

    cce = categorical_crossentropy(y_true, y_pred)
    weights = K.cast(tf.reduce_sum(class_weights*y_true),K.floatx())+K.epsilon()
    wcce = (weights * cce)/weights
    l1_norm = K.sum(K.abs(y_true - y_pred))+K.epsilon()
    l2_norm = K.sum(K.square(y_true - y_pred))+K.epsilon()
    return (lambda_0 * wcce) + (lambda_1 * l1_norm) + (lambda_2 * l2_norm)


################################################################################
# Implementation of the Focal loss.
# first proposed here: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
def _focal_loss(y_true, y_pred, alpha, is_loss):
    gamma = tf.constant(GAMMA, dtype=y_pred.dtype)
    axis_to_reduce = -1 if not is_TO_CATEG() else list(range(1, K.ndim(y_pred)))
    # Clip the prediction value to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    # Calculate Cross Entropy
    cross_entropy = -(y_true * K.log(y_pred))
    f_loss = alpha * K.pow((1 - y_pred), gamma) * cross_entropy
    return f_loss


def focal_loss(y_true, y_pred, is_loss=False):
    alpha = tf.constant(ALPHA, dtype=y_pred.dtype)
    return _focal_loss(y_true, y_pred, alpha, is_loss)


def focal_rest(y_true, y_pred, is_loss=False):
    alpha = tf.constant(ALPHA, dtype=y_pred.dtype)
    if get_n_classes() == 3: alpha = tf.constant([[0.25, 0, 0]], dtype=K.floatx())
    elif get_n_classes() == 2: alpha = tf.constant([[0.25, 0]], dtype=K.floatx())
    return _focal_loss(y_true, y_pred, alpha, is_loss)


def focal_p(y_true, y_pred, is_loss=False):
    alpha = tf.constant(ALPHA, dtype=y_pred.dtype)
    if get_n_classes() == 3: alpha = tf.constant([[0, 0.25, 0]], dtype=K.floatx())
    elif get_n_classes() == 2: alpha = tf.constant([[0, 0.25]], dtype=K.floatx())
    return _focal_loss(y_true, y_pred, alpha, is_loss)


def focal_c(y_true, y_pred, is_loss=False):
    alpha = tf.constant(ALPHA, dtype=y_pred.dtype)
    if get_n_classes() == 3: alpha = tf.constant([[0, 0, 0.25]], dtype=K.floatx())
    elif get_n_classes() == 2: alpha = tf.constant([[0, 0.25]], dtype=K.floatx())
    return _focal_loss(y_true, y_pred, alpha, is_loss)


################################################################################
# Function that computes the Tanimoto loss
def tanimoto(y_true, y_pred, is_loss, class_axis=None):
    """
    Compute weighted Tanimoto loss.
    Defined in the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data",
    under 3.2.4. Generalization to multiclass imbalanced problems. See https://arxiv.org/pdf/1904.00592.pdf
    """
    class_weights = general_utils.get_class_weights(is_loss=is_loss)
    y_true,y_pred,axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis, is_loss)

    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)
    denominator = K.square(y_true) + K.square(y_pred) - y_true * y_pred
    denominator = K.sum(denominator, axis=axis_to_reduce)
    out = numerator / (denominator+K.epsilon())
    return out


################################################################################
# Return precision as a metric
def prec_p(y_true, y_pred):
    class_weights = general_utils.get_class_weights()
    return _precision(y_true, y_pred, class_weights, class_axis=1)


def prec_c(y_true, y_pred):
    class_weights = general_utils.get_class_weights()
    return _precision(y_true, y_pred, class_weights, class_axis=get_n_classes()-1)


def _precision(y_true, y_pred, class_weights, class_axis):
    y_true, y_pred, axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis)

    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)
    denominator = K.sum(y_pred, axis=axis_to_reduce)
    out = numerator / (denominator + K.epsilon())
    return out


################################################################################
# Return recall as a metric
def rec_p(y_true, y_pred):
    class_weights = general_utils.get_class_weights()
    return _recall(y_true, y_pred, class_weights, class_axis=1)


def rec_c(y_true, y_pred):
    class_weights = general_utils.get_class_weights()
    return _recall(y_true, y_pred, class_weights, class_axis=get_n_classes()-1)


def _recall(y_true, y_pred, class_weights, class_axis):
    y_true, y_pred, axis_to_reduce = process_input(y_true, y_pred, class_weights, class_axis)

    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)
    denominator = K.sum(y_true, axis=axis_to_reduce)
    out = numerator / (denominator + K.epsilon())
    return out


################################################################################
# Return F1-score as a metric
def f1_p(y_true, y_pred):
    p = prec_p(y_true, y_pred)
    r = rec_p(y_true, y_pred)
    return 2. * ((p*r)/(p+r+K.epsilon()))


def f1_c(y_true, y_pred):
    p = prec_c(y_true, y_pred)
    r = rec_c(y_true, y_pred)
    return 2. * ((p*r)/(p+r+K.epsilon()))


################################################################################
# https://gist.github.com/Kautenja/69d306c587ccdf464c45d28c1545e580
def _iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 0
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 0.0, intersection / union)


def iou_p(y_true, y_pred): return _iou(y_true,y_pred,1)


def iou_c(y_true, y_pred): return _iou(y_true,y_pred,get_n_classes()-1)


def iou_rest(y_true, y_pred): return _iou(y_true,y_pred,0)


################################################################################
# Function to return the flatten couple (true/pred) or not, already multiplied by the class weight
def process_input(y_true, y_pred, class_weights, class_axis, is_loss=False):
    y_true *= class_weights
    y_pred *= class_weights
    if is_to_flat() and not is_loss:
        y_true = K.flatten(y_true) if class_axis is None else K.batch_flatten(y_true[..., class_axis])
        y_pred = K.flatten(y_pred) if class_axis is None else K.batch_flatten(y_pred[..., class_axis])
        axis_to_reduce = -1
    else: axis_to_reduce = -1 if not is_TO_CATEG() else list(range(1, K.ndim(y_pred)))  # All axis but first (batch)
    return y_true, y_pred, axis_to_reduce
