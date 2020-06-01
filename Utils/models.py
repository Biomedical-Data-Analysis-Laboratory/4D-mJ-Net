from Models import arch_mJNet, arch_van_De_Leemput, arch_Ronneberger_UNET


import constants
from Utils import general_utils, spatial_pyramid


from tensorflow.keras import layers, models, regularizers, initializers
import tensorflow.keras.backend as K

################################################################################
# mJ-Net model
def mJNet(X, params, to_categ):
    return arch_mJNet.mJNet(X, params, to_categ)
################################################################################
# Function to call the mJ-net with dropout
def mJNet_Drop(X, params, to_categ):
    return arch_mJNet.mJNet(X, params, to_categ, drop=True)

################################################################################
# Function to call the mJ-net with dropout and a long "J"
def mJNet_LongJ_Drop(X, params, to_categ):
    return arch_mJNet.mJNet(X, params, to_categ, drop=True, longJ=True)

################################################################################
# Function to call the mJ-net with a long "J"
def mJNet_LongJ(X, params, to_categ):
    return arch_mJNet.mJNet(X, params, to_categ, longJ=True)

################################################################################
# mJ-Net model version 2
def mJNet_v2(X, params, to_categ):
    return arch_mJNet.mJNet(X, params, to_categ, drop=True, longJ=False, v2=True)

################################################################################
# Model from Van De Leemput
def van_De_Leemput(X, params, to_categ):
    return arch_van_De_Leemput.van_De_Leemput(X, params, to_categ)

################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(X, params, to_categ):
    return arch_Ronneberger_UNET.Ronneberger_UNET(X, params, to_categ)
