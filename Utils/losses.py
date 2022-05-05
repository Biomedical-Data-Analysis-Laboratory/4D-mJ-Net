import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils import metrics
from Model.constants import *
import tensorflow.keras.backend as K
from tensorflow.keras import losses


################################################################################
# Function that calculates the modified DICE coefficient loss. Util for the LOSS
# function during the training of the model (for image in input and output)!
def squared_dice_coef_loss(y_true, y_pred):
    return 1-metrics.squared_dice_coef(y_true, y_pred, is_loss=True)


################################################################################
# Calculate the real value for the Dice coefficient, but it returns lower values than
# the other dice_coef + lower specificity and precision
def dice_coef_loss(y_true, y_pred):
    return 1-metrics.dice_coef(y_true, y_pred, is_loss=True)


################################################################################
# Tversky loss.
# Based on this paper: https://arxiv.org/abs/1706.05721
def tversky_loss(y_true, y_pred):
    return 1-metrics.tversky_coef(y_true, y_pred, is_loss=True)


################################################################################
# Focal Tversky loss: a generalisation of the tversky loss.
# From this paper: https://arxiv.org/abs/1810.07842
def focal_tversky_loss(y_true, y_pred):
    ft_params = get_Focal_Tversky()
    tv = metrics.tversky_coef(y_true, y_pred, is_loss=True)
    return K.pow((1 - tv), (1/ft_params["gamma"]))


################################################################################
# Function that calculates the JACCARD index loss. Util for the LOSS function during
# the training of the model (for image in input and output)!
def jaccard_index_loss(y_true, y_pred):
    return 1-metrics.jaccard_distance(y_true, y_pred)


################################################################################
# Function that calculate the weighted categorical crossentropy based on the
# article: https://doi.org/10.1109/ACCESS.2019.2910348
def weighted_categorical_cross_entropy_loss(y_true, y_pred):
    return metrics.weighted_categorical_cross_entropy(y_true, y_pred)


def weighted_categorical_crossentropy(weights):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        y_true = K.cast(y_true, y_pred.dtype)
        return losses.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce


################################################################################
#  Focal loss: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
def focal_loss(y_true, y_pred):
    return metrics.focal_loss(y_true, y_pred)


################################################################################
# Tanimoto loss. https://arxiv.org/pdf/1904.00592.pdf
def tanimoto_loss(y_true, y_pred):
    return 1-metrics.tanimoto(y_true, y_pred, is_loss=True)


################################################################################
# Tanimoto loss with its complement. https://arxiv.org/pdf/1904.00592.pdf
def tanimoto_with_dual_loss(y_true, y_pred):
    return 1-((metrics.tanimoto(y_true, y_pred, is_loss=True)+metrics.tanimoto(1-y_true, 1-y_pred, is_loss=True))/2)


################################################################################
# Hybrid loss containing the pixel wise cross-entropy and the soft dice coefficient
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932614
def pixelwise_crossentropy_plus_squared_dice_coeff(y_true, y_pred):
    cce = metrics.categorical_crossentropy(y_true, y_pred)
    sdc = metrics.squared_dice_coef(y_true, y_pred, is_loss=True)
    return -((cce + sdc) / K.cast(K.prod(K.shape(y_true)[:-1]), K.floatx()))


################################################################################
# Variant of the UNet++ loss
def pixelwise_crossentropy_plus_focal_tversky_loss(y_true, y_pred):
    tc = metrics.tversky_coef_hybrid(y_true, y_pred, is_loss=True)
    tc = tc/K.cast(K.prod(K.shape(y_true)[:-1]), K.floatx())
    return K.pow(1-tc, (1 / get_Focal_Tversky()["gamma"]))

