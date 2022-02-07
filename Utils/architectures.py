from Architectures import arch_mJNet, arch_PMs_segm, arch_autoencoder, arch_van_De_Leemput, arch_UNet, arch_VNet, arch_ResUNet, arch_mJNetplusplus, arch_TCN


################################################################################
# mJ-Net model
def mJNet(params, multi_input):
    return arch_mJNet.mJNet(params)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_NOBatch(params, multi_input):
    return arch_mJNet.mJNet(params, batch=False)


################################################################################
# Function to call the mJ-net with dropout
def mJNet_Drop(params, multi_input):
    return arch_mJNet.mJNet(params, drop=True)


################################################################################
# Function to call the mJ-net with dropout and a long "J"
def mJNet_LongJ_Drop(params, multi_input):
    return arch_mJNet.mJNet(params, drop=True, longJ=True)


################################################################################
# Function to call the mJ-net with a long "J"
def mJNet_LongJ(params, multi_input):
    return arch_mJNet.mJNet(params, longJ=True)


################################################################################
# Temporal Convolutional Network (TCN)
def TCNet(params, multi_input):
    return arch_TCN.TCNet(params)


################################################################################
# Temporal Convolutional Network (TCN)
def TCNet_single_encoder(params, multi_input):
    return arch_TCN.TCNet(params,single_enc=True)


################################################################################
# Temporal Convolutional Network (TCN)
def TCNet_3dot5D(params, multi_input):
    return arch_TCN.TCNet_3dot5D(params)


################################################################################
# Temporal Convolutional Network (TCN)
def TCNet_3dot5D_single_encoder(params, multi_input):
    return arch_TCN.TCNet_3dot5D(params, single_enc=True)

################################################################################
# mJ-Net model version 2
def mJNet_v2(params, multi_input):
    return arch_mJNet.mJNet(params, batch=True, drop=True, longJ=True, v2=True)


################################################################################
# mJ-Net model version 3D
def mJNet_2D_with_VGG16(params, multi_input):
    return arch_mJNet.mJNet_2D_with_VGG16(params, multi_input)


################################################################################
# mJ-Net model version 3D
def mJNet_3dot5D(params, multi_input, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_3dot5D(params, multi_input, usePMs=False, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net model version 3D
def mJNet_3dot5D_noDrop(params, multi_input, batch=True, drop=False, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_3dot5D(params, multi_input, usePMs=False, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net model version 3D
def mJNet_3dot5DWithPMS(params, multi_input, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_3dot5D(params, multi_input, usePMs=True, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net++ following this: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932614
def mJNet_plusplus(params, multi_input, batch=True, drop=True):
    return arch_mJNetplusplus.mJNet_plusplus(params, batch=batch, drop=drop)


################################################################################
# mJ-Net model version 4D
def mJNet_4D(params, multi_input, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_4D(params, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
# mJ-Net model version 4D
def mJNet_4D_xy(params, multi_input, batch=True, drop=True, leaky=True, attentiongate=True):
    return arch_mJNet.mJNet_4D_xy(params, batch=batch, drop=drop, leaky=leaky, attentiongate=attentiongate)


################################################################################
def PMs_segmentation(params, multi_input):
    return arch_PMs_segm.PMs_segmentation(params, multi_input)


################################################################################
def PMs_segmentation_NOBatch(params, multi_input):
    return arch_PMs_segm.PMs_segmentation(params, multi_input, batch=False)


################################################################################
def PMs_segmentation_multiInput(params, multi_input):
    return arch_PMs_segm.PMs_segmentation(params, multi_input)


################################################################################
def PMs_segmentation_multiInput_NOBatch(params, multi_input):
    return arch_PMs_segm.PMs_segmentation(params, multi_input, batch=False)


################################################################################
def simple_autoencoder(params, multi_input):
    return arch_autoencoder.simple_autoencoder(params)


################################################################################
# Model from Van De Leemput
def van_De_Leemput(params, multi_input):
    return arch_van_De_Leemput.van_De_Leemput(params)


################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(params, multi_input):
    return arch_UNet.Ronneberger_UNET(params)


################################################################################
# Model from Milletari V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params, multi_input):
    return arch_VNet.VNet_Milletari(params)


def VNet_Milletari_PMS(params, multi_input):
    return arch_VNet.VNet_Milletari_PMS(params, multi_input)


################################################################################
# Model for the V-Net Light https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098643
def VNet_Light(params, multi_input):
    return arch_VNet.VNet_Light(params)


################################################################################
# Model for the Res-UNet++ https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8959021
def ResUNet(params, multi_input):
    return arch_ResUNet.ResUNet(params)
