from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Conv2D, Conv3D, Concatenate, Conv2DTranspose, Conv3DTranspose, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16

import glob
from Utils import general_utils
from Model import constants


################################################################################
# Class that define a PM object
class PM_obj(object):
    def __init__(self, name, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
        self.name = ("_" + name)
        self.input_shape = (constants.getM(), constants.getN(), 3)

        # Create base model
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model._name += self.name
        for layer in self.base_model.layers: layer._name += self.name
        # Freeze base model
        self.base_model.trainable = False if params["trainable"]==0 else True
        self.input = self.base_model.input

        # Creating dictionary that maps layer names to the layers
        self.layer_dict = dict([(layer.name, layer) for layer in self.base_model.layers])

        # Conv layers after the VGG16
        self.conv_1 = Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                             kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                             kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.base_model.output)
        self.conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                             kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                             kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.conv_1)
        if batch: self.conv_2 = layers.BatchNormalization()(self.conv_2)
        self.conv_2 = Dropout(params["dropout"][name+".1"])(self.conv_2)


################################################################################
# Get the list of PMs classes
def getPMsList(multiInput, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
    PMS = []
    if "cbf" in multiInput.keys() and multiInput["cbf"] == 1:
        cbf = PM_obj("cbf", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(cbf)
    if "cbv" in multiInput.keys() and multiInput["cbv"] == 1:
        cbv = PM_obj("cbv", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(cbv)
    if "ttp" in multiInput.keys() and multiInput["ttp"] == 1:
        ttp = PM_obj("ttp", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(ttp)
    if "mtt" in multiInput.keys() and multiInput["mtt"] == 1:
        mtt = PM_obj("mtt", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(mtt)
    if "tmax" in multiInput.keys() and multiInput["tmax"] == 1:
        tmax = PM_obj("tmax", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(tmax)
    if "mip" in multiInput.keys() and multiInput["mip"]==1:
        mip = PM_obj("mip", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(mip)
    return PMS


################################################################################
# Get the correct regularizer
def getRegularizer(reg_obj):
    regularizer = None
    if reg_obj["type"]=="l1": regularizer = regularizers.l1(l=reg_obj["l"])
    elif reg_obj["type"]=="l2": regularizer = regularizers.l2(l=reg_obj["l"])
    elif reg_obj["type"]=="l1_l2": regularizer = regularizers.l1_l2(l1=reg_obj["l1"], l2=reg_obj["l2"])  # (l1=1e-6, l2=1e-5)
    return regularizer


################################################################################
# Function containing the transpose layers for the deconvolutional part
def upLayers(input, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
             bias_constraint, leaky=False, is2D=False):
    if is2D: transposeConv = Conv2DTranspose
    else: transposeConv = Conv3DTranspose

    conv = doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky, is2D=is2D)
    transp = transposeConv(channels[2], kernel_size=kernel_size, strides=strides_size, padding='same',
                           activation=activ_func, kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)
    if leaky: transp = layers.LeakyReLU(alpha=0.33)(transp)

    block_conc = Concatenate(-1)(block)
    return Concatenate(-1)([transp, block_conc])


################################################################################
# Check if the architecture need to have more info in the input
def addMoreInfo(multiInput, inputs, layersForAppending, is3D=False, is4D=False):
    # MORE INFO as input = NIHSS score, age, gender
    input_dim = 0
    concat_input = []
    flag_dense = 0

    if "nihss" in multiInput.keys() and multiInput["nihss"] == 1:
        NIHSS_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(NIHSS_input)
        flag_dense = 1
    if "age" in multiInput.keys() and multiInput["age"] == 1:
        age_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(age_input)
        flag_dense = 1
    if "gender" in multiInput.keys() and multiInput["gender"] == 1:
        gender_input = layers.Input(shape=(1,))
        input_dim += 1
        concat_input.append(gender_input)
        flag_dense = 1

    if flag_dense:
        if input_dim == 1: conc = concat_input[0]
        else: conc = Concatenate(1)(concat_input)
        dense_1 = layers.Dense(100, input_dim=input_dim, activation="relu")(conc)
        third_dim = 1 if not is3D else layersForAppending[0].shape[3]
        fourth_dim = 1 if not is3D else layersForAppending[0].shape[4]

        dense_2 = layers.Dense(layersForAppending[0].shape[1] * layersForAppending[0].shape[2] * third_dim * fourth_dim, activation="relu")(dense_1)
        out = layers.Reshape((layersForAppending[0].shape[1], layersForAppending[0].shape[2], third_dim))(dense_2)
        if is4D: out = layers.Reshape((layersForAppending[0].shape[1], layersForAppending[0].shape[2], third_dim, fourth_dim))(dense_2)
        multiInput_mdl = models.Model(inputs=concat_input, outputs=[out])
        inputs = [inputs, multiInput_mdl.input]
        layersForAppending.append(multiInput_mdl.output)

    return inputs, layersForAppending


################################################################################
# Function containing a block for the convolutional part
def blockConv3D(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
              leaky, batch, pool_size):
    conv_1 = Conv3D(channels[0], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input)
    if leaky: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    conv_1 = Conv3D(channels[1], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_1)
    if leaky: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    return layers.MaxPooling3D(pool_size)(conv_1)


################################################################################
# Function to compute two 3D convolutional layers
def doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                      bias_constraint, leaky, is2D=False):
    if is2D: convLayer = Conv2D
    else: convLayer = Conv3D

    conv = convLayer(channels[0], kernel_size=kernel_size, padding='same', activation=activ_func,
                     kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(input)
    if leaky: conv = layers.LeakyReLU(alpha=0.33)(conv)
    conv = convLayer(channels[1], kernel_size=kernel_size, padding='same', activation=activ_func,
                     kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint)(conv)
    if leaky: conv = layers.LeakyReLU(alpha=0.33)(conv)
    return conv


################################################################################
# Function to execute a double 3D convolution, followed by an attention gate, upsampling, and concatenation
def upSamplingPlusAttention(input, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init,
                            kernel_constraint, bias_constraint, leaky, is2D=False):
    conv = doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky, is2D)

    attGate = attentionGateBlock(x=block, g=conv, inter_shape=channels[2], l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, is2D=is2D)
    if is2D: up = layers.concatenate([layers.UpSampling2D(size=strides_size)(conv), attGate], axis=-1)
    else: up = layers.concatenate([layers.UpSampling3D(size=strides_size)(conv), attGate], axis=-1)
    return up


################################################################################
# Get the previous and next folder, given a specific folder and the slice index
def getPrevNextFolder(folder, sliceIndex):
    folders = []
    maxSlice = len(glob.glob(folder[:-3]+"*"))
    if int(sliceIndex)==1:
        folders.extend([folder,folder])
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)+1)+"/"))
    elif int(sliceIndex)==maxSlice:
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)-1)+"/"))
        folders.extend([folder, folder])
    else:
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)-1)+"/"))
        folders.append(folder)
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)+1)+"/"))

    return folders


################################################################################
# Anonymous lambda function to expand the specified axis by a factor of argument, rep.
# If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2
def expend_as(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)


################################################################################
# Attention gate block; from: https://arxiv.org/pdf/1804.03999.pdf
def attentionGateBlock(x, g, inter_shape, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=False):
    shape_x = tuple([i for i in list(K.int_shape(x[0])) if i])
    shape_g = tuple([i for i in list(K.int_shape(g[0])) if i])

    if is2D:
        convLayer = Conv2D
        upsampling = layers.UpSampling2D
        strides = (shape_x[0]//shape_g[0],shape_x[1]//shape_g[1])
    else:
        convLayer = Conv3D
        upsampling = layers.UpSampling3D
        strides = (shape_x[0]//shape_g[0],shape_x[1]//shape_g[1],shape_x[2]//shape_g[2])

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = convLayer(inter_shape, kernel_size=1, padding='same', kernel_regularizer=l1_l2_reg,
                      kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(g)
    # Getting the x signal to the same shape as the gating signal
    theta_x = convLayer(inter_shape, kernel_size=3, padding='same', kernel_regularizer=l1_l2_reg,
                        kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint, strides=strides)(x)
    # Element-wise addition of the gating and x signals
    add_xg = layers.Add()([phi_g, theta_x])
    add_xg = layers.Activation('relu')(add_xg)
    # 1x1x1 convolution
    psi = convLayer(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = layers.Activation('sigmoid')(psi)
    shape_sigmoid = tuple([i for i in list(K.int_shape(psi)) if i])
    if is2D: up_size = (shape_x[0]//shape_sigmoid[0],shape_x[1]//shape_sigmoid[1])
    else: up_size = (shape_x[0]//shape_sigmoid[0],shape_x[1]//shape_sigmoid[1],shape_x[2]//shape_sigmoid[2])
    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = upsampling(size=up_size)(psi)
    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[-1])
    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = layers.Multiply()([upsample_sigmoid_xg, x])
    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = convLayer(filters=shape_x[-1], kernel_size=1, padding='same')(attn_coefficients)
    output = layers.BatchNormalization()(output)
    return output
