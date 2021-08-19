from Model import constants
from Architectures import arch_mJNet, arch_PMs_segm, arch_autoencoder, arch_van_De_Leemput, arch_UNet, arch_VNet, arch_ResUNet, arch_mJNetplusplus
from Utils import general_utils


################################################################################
# mJ-Net model
def mJNet(params, multiInput):
    return arch_mJNet.mJNet(params)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_NOBatch(params, multiInput):
    return arch_mJNet.mJNet(params, batch=False)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_Drop(params, multiInput):
    return arch_mJNet.mJNet(params, drop=True)


################################################################################
# Function to call the mJ-net with dropout and a long "J"
def mJNet_LongJ_Drop(params, multiInput):
    return arch_mJNet.mJNet(params, drop=True, longJ=True)


################################################################################
# Function to call the mJ-net with a long "J"
def mJNet_LongJ(params, multiInput):
    return arch_mJNet.mJNet(params, longJ=True)


################################################################################
# mJ-Net model version 2
def mJNet_v2(params, multiInput):
    return arch_mJNet.mJNet(params, batch=True, drop=True, longJ=True, v2=True)


################################################################################
# mJ-Net model version 3D
def mJNet_2D_with_VGG16(params, multiInput):
    return arch_mJNet.mJNet_2D_with_VGG16(params, multiInput)


################################################################################
# mJ-Net model version 3D
def mJNet_4D(params, multiInput, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_4D(params, multiInput, usePMs=False, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net model version 3D
def mJNet_4DWithPMS(params, multiInput, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_4D(params, multiInput, usePMs=True, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net++ following this: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932614
def mJNet_plusplus(params, multiInput, batch=True, drop=True):
    return arch_mJNetplusplus.mJNet_plusplus(params, batch=batch, drop=drop)

################################################################################
def PMs_segmentation(params, multiInput):
    return arch_PMs_segm.PMs_segmentation(params, multiInput)


################################################################################
def PMs_segmentation_NOBatch(params, multiInput):
    return arch_PMs_segm.PMs_segmentation(params, multiInput, batch=False)


################################################################################
def PMs_segmentation_multiInput(params, multiInput):
    return arch_PMs_segm.PMs_segmentation(params, multiInput)


################################################################################
def PMs_segmentation_multiInput_NOBatch(params, multiInput):
    return arch_PMs_segm.PMs_segmentation(params, multiInput, batch=False)


################################################################################
def simple_autoencoder(params, multiInput):
    return arch_autoencoder.simple_autoencoder(params)


################################################################################
# Model from Van De Leemput
def van_De_Leemput(params, multiInput):
    return arch_van_De_Leemput.van_De_Leemput(params)


################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(params, multiInput):
    return arch_UNet.Ronneberger_UNET(params)


################################################################################
# Model from Milletari V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params, multiInput):
    return arch_VNet.VNet_Milletari(params)


def VNet_Milletari_PMS(params, multiInput):
    return arch_VNet.VNet_Milletari_PMS(params, multiInput)


################################################################################
# Model for the V-Net Light https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098643
def VNet_Light(params, multiInput):
    return arch_VNet.VNet_Light(params)


################################################################################
# Model for the Res-UNet++ https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8959021
def ResUNet(params, multiInput):
    return arch_ResUNet.ResUNet(params)
