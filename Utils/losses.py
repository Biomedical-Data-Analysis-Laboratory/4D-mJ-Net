from Utils import metrics

import tensorflow.keras.backend as K

################################################################################
# Function that calculates the modified DICE coefficient loss. Util for the LOSS function during the training of the model (for image in input and output)!
def mod_dice_coef_loss(y_true, y_pred):
    return 1-metrics.mod_dice_coef(y_true, y_pred)

################################################################################
# Calculate the real value for the Dice coefficient, but it returns lower values than the other dice_coef + lower specificity and precision
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
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1-metrics.dice_coef_binary(y_true, y_pred)

################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred, smooth=100):
    return (1-metrics.jaccard_distance(y_true, y_pred, smooth)) * smooth

################################################################################
# Function that calculate the weighted categorical crossentropy based on the
# article: https://doi.org/10.1109/ACCESS.2019.2910348
def weighted_categorical_cross_entropy_loss(y_true, y_pred):
    return metrics.weighted_categorical_cross_entropy(y_true, y_pred)
