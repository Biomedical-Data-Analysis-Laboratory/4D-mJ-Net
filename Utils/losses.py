from Utils import metrics

import tensorflow.keras.backend as K

################################################################################
# Function that calculates the modified DICE coefficient loss. Util for the LOSS function during the training of the model (for image in input and output)!
def mod_dice_coef_loss(y_true, y_pred):
    # return 1-metrics.mod_dice_coef(y_true, y_pred)
    return 1-K.mean(metrics.mod_dice_coef(y_true, y_pred), axis=-1)

################################################################################
# Calculate the real value for the Dice coefficient, but it returns lower values than the other dice_coef + lower specificity and precision
def dice_coef_loss(y_true, y_pred):
    # return 1-metrics.dice_coef(y_true, y_pred)
    return 1-K.mean(metrics.dice_coef(y_true, y_pred), axis=-1)

################################################################################
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

# TODO:
def generalized_dice_loss(y_true, y_pred):
    return 1-metrics.generalized_dice_coeff(y_true, y_pred)

# TODO:
def dice_coef_binary_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1-metrics.dice_coef_binary(y_true, y_pred)

################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred, smooth=1):
    return (1-metrics.jaccard_distance(y_true, y_pred, smooth)) * smooth

################################################################################
# Function that calculate the categorical crossentropy loss
def categorical_crossentropy_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

################################################################################
# Function that calculate the weighted categorical crossentropy based on the
# article: https://doi.org/10.1109/ACCESS.2019.2910348
def weighted_categorical_cross_entropy_loss(y_true, y_pred):
    return metrics.weighted_categorical_cross_entropy(y_true, y_pred)
