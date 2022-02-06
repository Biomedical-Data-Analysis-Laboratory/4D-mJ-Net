from Model.constants import *
from Utils import general_utils, model_utils

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K


################################################################################
# mJ-Net++ model that follows the idea from: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932614
def mJNet_plusplus(params, batch, drop):
    # The TimeDistributed layer works if the time dimension is on the first channel
    timedistr = not is_timelast()
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.get_regularizer(params["regularizer"])
    activ_func = None
    kernel_size = (3,3)
    size_two = (2,2)
    input_shape = (get_m(), get_n(), getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(),
                                                                                                 get_m(), get_n(), 1)
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.get_kernel_init(
        params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["bias_constraint"])
    channels = [16,32,64,64,128,256,512,1024]

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)
    time_conv_01 = model_utils.double_conv(input_x, [channels[0], channels[0]], kernel_size, activ_func, l1_l2_reg,
                                           kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                           timedistr=timedistr)
    if batch: time_conv_01 = layers.BatchNormalization()(time_conv_01)
    general_utils.print_int_shape(time_conv_01)
    pool_shape = (1,1,params["max_pool"]["long.1"]) if is_timelast() else (params["max_pool"]["long.1"], 1, 1)
    pool_drop_01 = layers.MaxPooling3D(pool_shape)(time_conv_01)
    general_utils.print_int_shape(pool_drop_01)

    time_conv_02 = model_utils.double_conv(pool_drop_01, [channels[1], channels[1]], kernel_size, activ_func, l1_l2_reg,
                                           kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                           timedistr=timedistr)
    if batch: time_conv_02 = layers.BatchNormalization()(time_conv_02)
    general_utils.print_int_shape(time_conv_02)
    pool_shape = (1,1,params["max_pool"]["long.2"]) if is_timelast() else (params["max_pool"]["long.2"], 1, 1)
    pool_drop_02 = layers.MaxPooling3D(pool_shape)(time_conv_02)
    general_utils.print_int_shape(pool_drop_02)

    time_conv_03 = model_utils.double_conv(pool_drop_02, [channels[2], channels[2]], kernel_size, activ_func, l1_l2_reg,
                                           kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                           timedistr=timedistr)
    if batch: time_conv_03 = layers.BatchNormalization()(time_conv_03)
    general_utils.print_int_shape(time_conv_03)
    pool_shape = (1,1,params["max_pool"]["long.3"]) if is_timelast() else (params["max_pool"]["long.3"], 1, 1)
    pool_drop_03 = layers.MaxPooling3D(pool_shape)(time_conv_03)
    general_utils.print_int_shape(pool_drop_03)
    if drop: pool_drop_03 = Dropout(params["dropout"]["long.1"])(pool_drop_03)

    # Here the time volume should become a 2D image
    pool_drop_03 = layers.Reshape((get_m(), get_n(), channels[2]))(pool_drop_03)
    general_utils.print_int_shape(pool_drop_03)
    conv_0_0 = model_utils.double_conv(pool_drop_03, [channels[3], channels[3]], kernel_size, activ_func, l1_l2_reg,
                                       kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                       timedistr=False)
    if batch: conv_0_0 = layers.BatchNormalization()(conv_0_0)
    general_utils.print_int_shape(conv_0_0)
    pool_drop_1 = layers.MaxPooling2D(size_two)(conv_0_0)
    general_utils.print_int_shape(pool_drop_1)

    conv_1_0 = model_utils.double_conv(pool_drop_1, [channels[4], channels[4]], kernel_size, activ_func, l1_l2_reg,
                                       kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                       timedistr=False)
    if batch: conv_1_0 = layers.BatchNormalization()(conv_1_0)
    general_utils.print_int_shape(conv_1_0)
    pool_drop_2 = layers.MaxPooling2D(size_two)(conv_1_0)
    general_utils.print_int_shape(pool_drop_2)

    conv_2_0 = model_utils.double_conv(pool_drop_2, [channels[5], channels[5]], kernel_size, activ_func, l1_l2_reg,
                                       kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                       timedistr=False)
    if batch: conv_2_0 = layers.BatchNormalization()(conv_2_0)
    general_utils.print_int_shape(conv_2_0)
    pool_drop_3 = layers.MaxPooling2D(size_two)(conv_2_0)
    general_utils.print_int_shape(pool_drop_3)

    conv_3_0 = model_utils.double_conv(pool_drop_3, [channels[6], channels[6]], kernel_size, activ_func, l1_l2_reg,
                                       kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                       timedistr=False)
    if batch: conv_3_0 = layers.BatchNormalization()(conv_3_0)
    general_utils.print_int_shape(conv_3_0)
    pool_drop_4 = layers.MaxPooling2D(size_two)(conv_3_0)
    general_utils.print_int_shape(pool_drop_4)

    conv_4_0 = model_utils.double_conv(pool_drop_4, [channels[7], channels[7]], kernel_size, activ_func, l1_l2_reg,
                                       kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                       timedistr=False)
    if batch: conv_4_0 = layers.BatchNormalization()(conv_4_0)
    general_utils.print_int_shape(conv_4_0)
    if drop: conv_4_0 = Dropout(params["dropout"]["5"])(conv_4_0)

    transp_conv_4_0 = Conv2DTranspose(channels[6], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4_0)
    up_3_1 = Concatenate(-1)([transp_conv_4_0, conv_3_0])
    general_utils.print_int_shape(up_3_1)
    up_3_1 = model_utils.double_conv(up_3_1, [channels[6], channels[6]], kernel_size, activ_func, l1_l2_reg,
                                     kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                     timedistr=False)
    if batch: up_3_1 = layers.BatchNormalization()(up_3_1)

    transp_conv_3_0 = Conv2DTranspose(channels[5], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_3_0)
    up_2_1 = Concatenate(-1)([transp_conv_3_0, conv_2_0])
    general_utils.print_int_shape(up_2_1)
    up_2_1 = model_utils.convolution_layer(up_2_1, channels[5], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_2_1 = layers.BatchNormalization()(up_2_1)

    transp_conv_2_0 = Conv2DTranspose(channels[4], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_2_0)
    up_1_1 = Concatenate(-1)([transp_conv_2_0, conv_1_0])
    general_utils.print_int_shape(up_1_1)
    up_1_1 = model_utils.convolution_layer(up_1_1, channels[4], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_1_1 = layers.BatchNormalization()(up_1_1)

    transp_conv_1_0 = Conv2DTranspose(channels[3], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_1_0)
    up_0_1 = Concatenate(-1)([transp_conv_1_0, conv_0_0])
    general_utils.print_int_shape(up_0_1)
    up_0_1 = model_utils.convolution_layer(up_0_1, channels[3], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_0_1 = layers.BatchNormalization()(up_0_1)

    transp_up_3_1 = Conv2DTranspose(channels[4], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_3_1)
    up_2_2 = Concatenate(-1)([transp_up_3_1, up_2_1, conv_2_0])
    general_utils.print_int_shape(up_2_2)
    up_2_2 = model_utils.double_conv(up_2_2, [channels[4], channels[4]], kernel_size, activ_func, l1_l2_reg,
                                     kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                     timedistr=False)
    if batch: up_2_2 = layers.BatchNormalization()(up_2_2)

    transp_up_2_1 = Conv2DTranspose(channels[3], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_2_1)
    up_1_2 = Concatenate(-1)([transp_up_2_1, up_1_1, conv_1_0])
    up_1_2 = model_utils.convolution_layer(up_1_2, channels[3], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_1_2 = layers.BatchNormalization()(up_1_2)

    transp_up_1_1 = Conv2DTranspose(channels[2], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_1_1)
    up_0_2 = Concatenate(-1)([transp_up_1_1, up_0_1, conv_0_0])
    up_0_2 = model_utils.convolution_layer(up_0_2, channels[2], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_0_2 = layers.BatchNormalization()(up_0_2)

    transp_up_2_2 = Conv2DTranspose(channels[2], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_2_2)
    up_1_3 = Concatenate(-1)([transp_up_2_2, up_1_2, up_1_1, conv_1_0])
    up_1_3 = model_utils.double_conv(up_1_3, [channels[2], channels[2]], kernel_size, activ_func, l1_l2_reg,
                                     kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True,
                                     timedistr=False)
    if batch: up_1_3 = layers.BatchNormalization()(up_1_3)

    transp_up_1_2 = Conv2DTranspose(channels[1], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_1_2)
    up_0_3 = Concatenate(-1)([transp_up_1_2, up_0_2, up_0_1, conv_0_0])
    up_0_3 = model_utils.convolution_layer(up_0_3, channels[1], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, leaky=True, is2D=True, timedistr=False)
    if batch: up_0_3 = layers.BatchNormalization()(up_0_3)

    # set the softmax activation function if the flag is set
    act_name = "softmax" if is_TO_CATEG() else "sigmoid"
    n_chann = len(get_labels()) if is_TO_CATEG() else 1
    shape_output = (get_m(), get_n(), n_chann) if is_TO_CATEG() else (get_m(), get_n())

    transp_up_1_3 = Conv2DTranspose(channels[0], kernel_size=kernel_size, strides=size_two, activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_1_3)
    up_0_4 = Concatenate(-1)([transp_up_1_3, up_0_3, up_0_2, up_0_1, conv_0_0])
    up_0_4 = model_utils.convolution_layer(up_0_4, n_chann, (1, 1), act_name, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, is2D=True)
    general_utils.print_int_shape(up_0_4)
    y_0_4 = layers.Reshape(shape_output)(up_0_4)

    up_0_3 = model_utils.convolution_layer(up_0_3, n_chann, (1, 1), act_name, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, is2D=True)
    general_utils.print_int_shape(up_0_3)
    y_0_3 = layers.Reshape(shape_output)(up_0_3)

    up_0_2 = model_utils.convolution_layer(up_0_2, n_chann, (1, 1), act_name, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, is2D=True)
    general_utils.print_int_shape(up_0_2)
    y_0_2 = layers.Reshape(shape_output)(up_0_2)

    up_0_1 = model_utils.convolution_layer(up_0_1, n_chann, (1, 1), act_name, l1_l2_reg, kernel_init, 'same',
                                           kernel_constraint, bias_constraint, is2D=True)
    general_utils.print_int_shape(up_0_1)
    y_0_1 = layers.Reshape(shape_output)(up_0_1)

    y = layers.Average()([y_0_1,y_0_2,y_0_3,y_0_4])

    model = models.Model(inputs=input_x, outputs=y)
    return model
