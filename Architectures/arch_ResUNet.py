from Model.constants import *
from Utils import general_utils, model_utils

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv3D, Conv2DTranspose, Conv3DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K


################################################################################
# Res-UNet++ model https://github.com/DebeshJha/ResUNetPlusPlus-with-CRF-and-TTA
def ResUNet(params):
    channels = [32,64,128,256,512]
    channels = [int(ch/8) for ch in channels]

    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.get_regularizer(params["regularizer"])
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.get_kernel_init(
        params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["bias_constraint"])
    input_shape = (get_m(), get_n(), getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(),
                                                                                                 get_m(), get_n(), 1)

    input_x = layers.Input(shape=input_shape, sparse=False)
    conv_1 = Conv3D(channels[0], kernel_size=(3,3,3),kernel_regularizer=l1_l2_reg,padding="same",kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)

    conv_2 = Conv3D(channels[0], kernel_size=(1,1,1),kernel_regularizer=l1_l2_reg,padding="same",kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
    conv_2 = layers.BatchNormalization()(conv_2)

    add_1 = layers.Add()([conv_1, conv_2])
    squeeze_exc_1 = model_utils.squeeze_excite_block(add_1)
    general_utils.print_int_shape(squeeze_exc_1)

    stride = (2,2,params["strides"]["1"]) if is_timelast() else (params["strides"]["1"], 2, 2)
    resblock_1 = model_utils.block_resNet(squeeze_exc_1, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[1], strides=stride)
    general_utils.print_int_shape(resblock_1)
    stride = (2,2,params["strides"]["2"]) if is_timelast() else (params["strides"]["2"], 2, 2)
    resblock_2 = model_utils.block_resNet(resblock_1, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[2], strides=stride)
    general_utils.print_int_shape(resblock_2)
    stride = (2,2,params["strides"]["3"]) if is_timelast() else (params["strides"]["3"], 2, 2)
    resblock_3 = model_utils.block_resNet(resblock_2, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[3], strides=stride)
    general_utils.print_int_shape(resblock_3)

    bridge = model_utils.ASSP(resblock_3, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, filt=channels[4])
    general_utils.print_int_shape(bridge)

    att_1 = model_utils.block_attentionGate(resblock_2, bridge, channels[4], l1_l2_reg, kernel_init, kernel_constraint,
                                            bias_constraint)
    general_utils.print_int_shape(att_1)
    resblock_4 = model_utils.block_resNet(att_1, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[3], strides=(1, 1, 1))
    general_utils.print_int_shape(resblock_4)

    att_2 = model_utils.block_attentionGate(resblock_1, resblock_4, channels[3], l1_l2_reg, kernel_init,
                                            kernel_constraint, bias_constraint)
    general_utils.print_int_shape(att_2)

    resblock_5 = model_utils.block_resNet(att_2, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[2], strides=(1, 1, 1))
    general_utils.print_int_shape(resblock_5)

    att_3 = model_utils.block_attentionGate(squeeze_exc_1, resblock_5, channels[2], l1_l2_reg, kernel_init,
                                            kernel_constraint, bias_constraint)
    general_utils.print_int_shape(att_3)

    resblock_6 = model_utils.block_resNet(att_3, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                          filters=channels[1], strides=(1, 1, 1))
    general_utils.print_int_shape(resblock_6)

    assp_block = model_utils.ASSP(resblock_6, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                  filt=channels[0])
    general_utils.print_int_shape(assp_block)
    act_name = "sigmoid"
    n_chann = 1
    shape_output = (get_m(), get_n())

    # set the softmax activation function if the flag is set
    if is_TO_CATEG():
        act_name = "softmax"
        n_chann = len(get_labels())
        shape_output = (get_m(), get_n(), n_chann)

    stride = (1, 1, params["strides"]["1"]) if is_timelast() else (params["strides"]["1"], 1, 1)
    last_conv = layers.Conv3D(channels[0], kernel_size=(3,3,3), activation="relu", padding='same', kernel_initializer=kernel_init, strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(assp_block)
    stride = (1, 1, params["strides"]["2"]) if is_timelast() else (params["strides"]["2"], 1, 1)
    last_conv = layers.Conv3D(channels[0]*2, kernel_size=(3,3,3), activation="relu", padding='same', kernel_initializer=kernel_init, strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    stride = (1, 1, params["strides"]["3"]) if is_timelast() else (params["strides"]["3"], 1, 1)
    last_conv = layers.Conv3D(channels[0]*4, kernel_size=(3,3,3), activation="relu", padding='same', kernel_initializer=kernel_init, strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)

    output = Conv3D(n_chann,kernel_size=(1,1,1),kernel_regularizer=l1_l2_reg,padding="same",kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    output = layers.Activation(act_name)(output)
    output = layers.Reshape(shape_output)(output)

    model = models.Model(inputs=input_x, outputs=output)

    return model
