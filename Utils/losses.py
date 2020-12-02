import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import metrics
import numpy as np
import tensorflow.keras.backend as K


################################################################################
# Function that calculates the modified DICE coefficient loss. Util for the LOSS
# function during the training of the model (for image in input and output)!
def mod_dice_coef_loss(y_true, y_pred):
    return 1-metrics.mod_dice_coef(y_true, y_pred)


################################################################################
# Calculate the real value for the Dice coefficient, but it returns lower values than
# the other dice_coef + lower specificity and precision
def dice_coef_loss(y_true, y_pred):
    return 1-metrics.dice_coef(y_true, y_pred)


################################################################################
# Tversky loss.
# Based on this paper: https://arxiv.org/abs/1706.05721
def tversky_loss(y_true, y_pred):
    return 1-metrics.tversky(y_true, y_pred)


################################################################################
# Focal Tversky loss: a generalisation of the tversky loss.
# From this paper: https://arxiv.org/abs/1810.07842
def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = metrics.tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


################################################################################
# TODO: implement
def generalized_dice_loss(y_true, y_pred):
    return 1-metrics.generalized_dice_coeff(y_true, y_pred)


################################################################################
# TODO: implement
def dice_coef_binary_loss(y_true, y_pred):
    """
    Dice loss to minimize. Pass to model as loss during compile statement
    """
    return 1-metrics.dice_coef_binary(y_true, y_pred)


################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during
# the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred, smooth=100):
    return (1-metrics.jaccard_distance(y_true, y_pred, smooth)) * smooth


################################################################################
# Function that calculate the weighted categorical crossentropy based on the
# article: https://doi.org/10.1109/ACCESS.2019.2910348
def weighted_categorical_cross_entropy_loss(y_true, y_pred):
    return metrics.weighted_categorical_cross_entropy(y_true, y_pred)


################################################################################
################################################################################
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]],gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        ret = K.sum(loss, axis=-1)
        #ret = K.print_tensor(ret, message='y_true = ')
        return ret

    return categorical_focal_loss_fixed
