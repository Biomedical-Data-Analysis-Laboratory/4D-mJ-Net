from Model import constants
from Utils import general_utils, spatial_pyramid, model_utils

from tensorflow.keras import layers, models, initializers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv3D, Conv2DTranspose, Conv3DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16


################################################################################
# mJ-Net model
def mJNet(params, to_categ, batch=True, drop=False, longJ=False, v2=False):
    # from (30,M,N) to (1,M,N)

    size_two = (2,2,1)  # (1,2,2)
    kernel_size = (3,3,1)
    activ_func = 'relu'
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    channels = [16,32,16,32,16,32,16,32,64,64,128,128,256,-1,-1,-1,-1,128,128,64,64,32,16]
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.
    kernel_constraint, bias_constraint = None, None  # max_norm(2.), max_norm(2.)

    if v2:  # version 2
        # size_two = (2,2,1)
        activ_func = None
        # Hu initializer
        kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
        kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

        # channels = [16,32,32,64,64,128,128,32,64,128,256,512,1024,512,1024,512,1024,-1,512,256,-1,128,64]
        channels = [16,16,32,32,64,64,-1,64,64,128,128,128,128,128,128,128,128,-1,128,128,-1,64,32]
        channels = [int(ch/2) for ch in channels]  # implemented due to memory issues

        # input_shape = (None,constants.getM(),constants.getN(),1)
        # TODO: input_shape = (constants.NUMBER_OF_IMAGE_PER_SECTION,None,None,1)

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)  # (None, 30, M, N, 1)

    if longJ:
        conv_01 = Conv3D(channels[0], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        if batch: conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 16)
        conv_01 = Conv3D(channels[1], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_01)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        if batch: conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 32)

        # pool_drop_01 = layers.MaxPooling3D((params["max_pool"]["long.1"],1,1))(conv_01)
        pool_drop_01 = layers.MaxPooling3D((1,1,params["max_pool"]["long.1"]))(conv_01)
        general_utils.print_int_shape(pool_drop_01)  # (None, 15, M, N, 32)
        conv_02 = Conv3D(channels[2], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_01)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        if batch: conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 32)
        conv_02 = Conv3D(channels[3], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_02)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        if batch: conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 64)

        # pool_drop_02 = layers.MaxPooling3D((params["max_pool"]["long.2"],1,1))(conv_02)
        pool_drop_02 = layers.MaxPooling3D((1,1,params["max_pool"]["long.2"]))(conv_02)
        general_utils.print_int_shape(pool_drop_02)  # (None, 5, M, N, 64)
        conv_03 = Conv3D(channels[4], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_02)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        if batch: conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 64)
        conv_03 = Conv3D(channels[5], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_03)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        if batch: conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 128)

        # pool_drop_1 = layers.MaxPooling3D((params["max_pool"]["long.3"],1,1))(conv_03)
        pool_drop_1 = layers.MaxPooling3D((1,1,params["max_pool"]["long.3"]))(conv_03)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["long.1"])(pool_drop_1)
    else:
        # conv_1 = Conv3D(channels[6], kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(input_x)
        conv_1 = Conv3D(channels[6], kernel_size=(3, 3, constants.NUMBER_OF_IMAGE_PER_SECTION), activation=activ_func,
                        padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
        if v2: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
        if batch: conv_1 = layers.BatchNormalization()(conv_1)
        general_utils.print_int_shape(conv_1)  # (None, 30, M, N, 128)
        # TODO: make this dynamic based on the original flag
        # pool_drop_1 = layers.AveragePooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(conv_1)
        pool_drop_1 = layers.AveragePooling3D((1, 1, constants.NUMBER_OF_IMAGE_PER_SECTION))(conv_1)
        # pool_drop_1 = spatial_pyramid.SPP3D([1,2,4], input_shape=(channels[6],None,None,None))(conv_1)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (1,M,N) to (1,M/2,N/2)
    conv_2 = Conv3D(channels[7], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_1)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, 1, M, N, 32)
    conv_2 = Conv3D(channels[8], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_2)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, 1, M, N, 64)
    pool_drop_2 = layers.MaxPooling3D(size_two)(conv_2)
    general_utils.print_int_shape(pool_drop_2)  # (None, 1, M/2, N/2, 64)
    if drop: pool_drop_2 = Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (1,M/2,N/2) to (1,M/4,N/4)
    conv_3 = Conv3D(channels[9], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_2)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, 1, M/2, N/2, 128)
    conv_3 = Conv3D(channels[10], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_3)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, 1, M/2, N/2, 256)
    pool_drop_3 = layers.MaxPooling3D(size_two)(conv_3)
    general_utils.print_int_shape(pool_drop_3)  # (None, 1, M/4, N/4, 256)
    if drop: pool_drop_3 = Dropout(params["dropout"]["3"])(pool_drop_3)

    # from (1,M/4,N/4) to (1,M/8,N/8)
    conv_4 = Conv3D(channels[11], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_3)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 512)
    conv_4 = Conv3D(channels[12], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 1024)

    if v2:
        pool_drop_3_1 = layers.MaxPooling3D(size_two)(conv_4)
        general_utils.print_int_shape(pool_drop_3_1)  # (None, 1, M/8, N/8, 1024)
        conv_4_1 = Conv3D(channels[13], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_3_1)
        conv_4_1 = layers.LeakyReLU(alpha=0.33)(conv_4_1)
        if batch: conv_4_1 = layers.BatchNormalization()(conv_4_1)
        general_utils.print_int_shape(conv_4_1)  # (None, 1, M/8, N/8, 512)
        conv_5_1 = Conv3D(channels[14], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4_1)
        conv_5_1 = layers.LeakyReLU(alpha=0.33)(conv_5_1)
        if batch: conv_5_1 = layers.BatchNormalization()(conv_5_1)
        if drop: conv_5_1 = Dropout(params["dropout"]["3.1"])(conv_5_1)
        general_utils.print_int_shape(conv_5_1)  # (None, 1, M/8, N/8, 1024)
        # add_1 = layers.add([pool_drop_3_1, conv_5_1])
        # general_utils.print_int_shape(add_1)  # (None, 1, M/8, N/8, 1024)
        # up_01 = layers.UpSampling3D(size=size_two)(add_1)
        # general_utils.print_int_shape(up_01)  # (None, 1, M/4, N/4, 1024)
        # conc_1 = layers.concatenate([up_01, conv_4], axis=-1)
        # general_utils.print_int_shape(conc_1)  # (None, 1, M/4, N/4, 1024)

        attGate_1 = model_utils.attentionGateBlock(x=conv_4, g=conv_5_1, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_0 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_5_1), attGate_1], axis=-1)

        conv_6_1 = Conv3D(channels[15], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_0)
        conv_6_1 = layers.LeakyReLU(alpha=0.33)(conv_6_1)
        if batch: conv_6_1 = layers.BatchNormalization()(conv_6_1)
        general_utils.print_int_shape(conv_6_1)  # (None, 1, M/4, N/4, 512)
        conv_7_1 = Conv3D(channels[16], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_6_1)
        conv_7_1 = layers.LeakyReLU(alpha=0.33)(conv_7_1)
        if batch: conv_7_1 = layers.BatchNormalization()(conv_7_1)
        general_utils.print_int_shape(conv_7_1)  # (None, 1, M/4, N/4, 1024)
        # add_2 = layers.add([conv_4, conv_7_1])
        # general_utils.print_int_shape(add_2)  # (None, 1, M/4, N/4, 1024)
        # up_02 = layers.UpSampling3D(size=size_two)(add_2)
        # general_utils.print_int_shape(up_02)  # (None, 1, M/2, N/2, 1024)
        # up_1 = layers.concatenate([up_02, conv_3])

        attGate_2 = model_utils.attentionGateBlock(x=conv_3, g=conv_7_1, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_1 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_7_1), attGate_2], axis=-1)
    else:
        # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
        up_1 = layers.concatenate([
            Conv3DTranspose(channels[17], kernel_size=size_two, strides=size_two, activation=activ_func,
                            padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4),
            conv_3], axis=3)

    general_utils.print_int_shape(up_1)  # (None, 1, M/2, N/2, 1024)
    conv_5 = Conv3D(channels[18], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_1)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    if batch: conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, 1, M/2, N/2, 512)
    conv_5 = Conv3D(channels[19], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_5)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    if batch: conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, 1, M/2, N/2, 256)

    if v2:
        attGate_3 = model_utils.attentionGateBlock(x=conv_2, g=conv_5, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_2 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_5), attGate_3], axis=-1)


        # if batch: addconv_5 = layers.concatenate([conv_5, conv_5])
        # while K.int_shape(addconv_5)[-1] != K.int_shape(up_1)[-1]:
        #     addconv_5 = layers.concatenate([addconv_5, addconv_5])
        # add_3 = layers.add([up_1, addconv_5])
        # general_utils.print_int_shape(add_3)  # (None, 1, M/2, N/4, 1024)
        # up_03 = layers.UpSampling3D(size=size_two)(add_3)
        # general_utils.print_int_shape(up_03)  # (None, 1, M, N, 1024)
        #
        # addconv_2 = layers.concatenate([conv_2, conv_2])
        # while K.int_shape(addconv_2)[-1] != K.int_shape(up_03)[-1]:
        #     addconv_2 = layers.concatenate([addconv_2, addconv_2])
        # up_2 = layers.concatenate([up_03, addconv_2])
    else:
        pool_drop_4 = layers.MaxPooling3D((1,1,2))(conv_5)
        general_utils.print_int_shape(pool_drop_4)  # (None, 1, M/2, N/2, 512)
        if drop: pool_drop_4 = Dropout(params["dropout"]["4"])(pool_drop_4)
        # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
        up_2 = layers.concatenate([
            Conv3DTranspose(channels[20], kernel_size=size_two, strides=size_two, activation=activ_func,
                            padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_4),
            conv_2], axis=3)

    general_utils.print_int_shape(up_2)  # (None, X, M, N, 1024)
    conv_6 = Conv3D(channels[21], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_2)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    if batch: conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6)  # (None, X, M, N, 128)
    conv_6 = Conv3D(channels[22], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_6)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    pool_drop_5 = layers.BatchNormalization()(conv_6) if batch else conv_6
    general_utils.print_int_shape(pool_drop_5)  # (None, X, M, N, 64)

    if not v2:
        # from (2,M,N)  to (1,M,N)
        pool_drop_5 = layers.MaxPooling3D((1,1,2))(pool_drop_5)
        general_utils.print_int_shape(pool_drop_5)  # (None, 1, M, N, 16)
        if drop: pool_drop_5 = Dropout(params["dropout"]["5"])(pool_drop_5)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = Conv3D(n_chann, (1,1,1), activation=act_name, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_5)
    general_utils.print_int_shape(conv_7)  # (None, 1, M, N, 1)
    y = layers.Reshape(shape_output)(conv_7)
    general_utils.print_int_shape(y)  # (None, M, N)
    model = models.Model(inputs=input_x, outputs=y)

    return model


################################################################################
# mJ-Net model version 3D ?
def mJNet_2D_with_VGG16(params, to_categ, multiInput, batch=True, drop=True, leaky=True, attentiongate=True):
    kernel_size, size_two = (3,3), (2,2)
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    activ_func = None if leaky else 'relu'
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)  # Hu initializer

    x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(x)

    conv_1 = layers.Conv3D(32, kernel_size=(3,3,constants.NUMBER_OF_IMAGE_PER_SECTION), activation=activ_func,
                           kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(x)
    if leaky: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)

    conv_1 = layers.Conv3D(32, kernel_size=(3,3,constants.NUMBER_OF_IMAGE_PER_SECTION), activation=activ_func,
                           kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                           bias_constraint=bias_constraint, strides=(1,1,params["strides"]["conv.1"]),
                           kernel_constraint=kernel_constraint)(conv_1)
    if leaky: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)

    new_z = constants.NUMBER_OF_IMAGE_PER_SECTION/params["strides"]["conv.1"]
    conv_2 = layers.Conv3D(32, kernel_size=(3,3,int(new_z)), activation=activ_func, kernel_constraint=kernel_constraint,
                           kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                           bias_constraint=bias_constraint)(conv_1)
    if leaky: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)

    conv_2 = layers.Conv3D(16, kernel_size=(3,3,int(new_z)), activation=activ_func, kernel_regularizer=l1_l2_reg,
                           kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, padding='same',
                           bias_constraint=bias_constraint, strides=(1,1,params["strides"]["conv.2"]))(conv_2)
    if leaky: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)

    new_z /= params["strides"]["conv.2"]
    conv_3 = layers.Conv3D(8, kernel_size=(3,3,int(new_z)), activation=activ_func, kernel_regularizer=l1_l2_reg,
                           kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, padding='same',
                           bias_constraint=bias_constraint)(conv_2)
    if leaky: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)

    conv_3 = layers.Conv3D(1, kernel_size=(3,3,int(new_z)), activation=activ_func, kernel_regularizer=l1_l2_reg,
                           kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, padding='same',
                           bias_constraint=bias_constraint, strides=(1,1,params["strides"]["conv.3"]))(conv_3)
    if leaky: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)

    conv_3 = layers.Reshape(conv_3.shape[1:-1])(conv_3)
    general_utils.print_int_shape(conv_3)

    if conv_3.shape[-1]==1: conv_3 = Concatenate(-1)([conv_3,conv_3,conv_3])
    general_utils.print_int_shape(conv_3)
    # Add the VGG-16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=conv_3.shape[1:])
    # Freeze layers VGG-16 model
    vgg16_model.trainable = False if params["trainable"] == 0 else True
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg16_model.layers])

    conv_4 = layers.Conv2D(128, kernel_size=kernel_size, padding='same',activation=activ_func, kernel_regularizer=l1_l2_reg,
                           kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint)(vgg16_model.output)
    if leaky: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    conv_4 = layers.Conv2D(128, kernel_size=kernel_size, padding='same', activation=activ_func, kernel_regularizer=l1_l2_reg,
                           kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint)(conv_4)
    if leaky: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    if drop: conv_4 = Dropout(params["dropout"]["conv.4"])(conv_4)

    inputs, conv_out = model_utils.addMoreInfo(multiInput, x, [conv_4])
    if len(conv_out)>1: conv_out = Concatenate(-1)(conv_out)
    elif len(conv_out)==1: conv_out = conv_out[0]

    if attentiongate:
        attGate_1 = model_utils.attentionGateBlock(x=layer_dict["block5_conv3"].output, g=conv_out, inter_shape=256, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, is2D=True)
        up_1 = layers.concatenate([layers.UpSampling2D(size=size_two)(conv_out), attGate_1], axis=-1)
        up_2 = model_utils.upSamplingPlusAttention(up_1,layer_dict["block4_conv3"].output,[128,128,128],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_3 = model_utils.upSamplingPlusAttention(up_2,layer_dict["block3_conv3"].output,[64,64,64],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_4 = model_utils.upSamplingPlusAttention(up_3,layer_dict["block2_conv2"].output,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_5 = model_utils.upSamplingPlusAttention(up_4,layer_dict["block1_conv2"].output,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
    else:
        transp_1 = Conv2DTranspose(256, kernel_size=size_two, strides=size_two, activation=activ_func, padding='same',
                                   kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_out)
        if leaky: transp_1 = layers.LeakyReLU(alpha=0.33)(transp_1)
        if batch: transp_1 = layers.BatchNormalization()(transp_1)
        up_1 = Concatenate(-1)([transp_1, layer_dict["block5_conv3"].output])
        # going up with the layers
        up_2 = model_utils.upLayers(up_1, layer_dict["block4_conv3"].output, [128,128,128], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_3 = model_utils.upLayers(up_2, layer_dict["block3_conv3"].output, [64,64,64], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_4 = model_utils.upLayers(up_3, layer_dict["block2_conv2"].output, [32,32,32], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_5 = model_utils.upLayers(up_4, layer_dict["block1_conv2"].output, [16,16,16], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)

    general_utils.print_int_shape(up_1)
    general_utils.print_int_shape(up_2)
    general_utils.print_int_shape(up_3)
    general_utils.print_int_shape(up_4)
    general_utils.print_int_shape(up_5)

    final_conv_1 = layers.Conv2D(16, kernel_size=kernel_size, padding='same', activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_5)
    if leaky: final_conv_1 = layers.LeakyReLU(alpha=0.33)(final_conv_1)
    if batch: final_conv_1 = layers.BatchNormalization()(final_conv_1)
    general_utils.print_int_shape(final_conv_1)
    final_conv_2 = layers.Conv2D(16, kernel_size=kernel_size, padding='same', activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_1)
    if leaky: final_conv_2 = layers.LeakyReLU(alpha=0.33)(final_conv_2)
    if batch: final_conv_2 = layers.BatchNormalization()(final_conv_2)
    general_utils.print_int_shape(final_conv_2)
    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    conv_last = layers.Conv2D(n_chann, (1,1), activation=act_name, padding='same', kernel_regularizer=l1_l2_reg,
                              kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_2)
    general_utils.print_int_shape(conv_last)
    y = layers.Reshape(shape_output)(conv_last)
    general_utils.print_int_shape(y)

    model_base = models.Model(vgg16_model.input, y)

    model = models.Model(inputs=inputs, outputs=model_base(conv_3))

    return model


################################################################################
# mJ-Net model with 4D data as input
def mJNet_4D(params, to_categ, multiInput, usePMs=True, batch=True, drop=False, leaky=True, attentiongate=True):
    size_two = (2,2,1)
    kernel_size, kernel_size_up = (3,3,1), (3,3,3)
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    activ_func = None if leaky else 'relu'
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)
    n_slices = 0 if "n_slices" not in params.keys() else params["n_slices"]
    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

    conv_out, block_6, block_5, block_4, block_3, inputs = [], [], [], [], [], []
    for slice in range(1,n_slices+1):
        input_x = layers.Input(shape=input_shape, sparse=False)
        inputs.append(input_x)

        out_1 = model_utils.blockConv3D(input_x,[8,8],(3,3,constants.NUMBER_OF_IMAGE_PER_SECTION),activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,(1,1,params["max_pool"][str(slice)+".long.1"]))
        new_z = int(constants.NUMBER_OF_IMAGE_PER_SECTION/params["max_pool"][str(slice)+".long.1"])
        out_2 = model_utils.blockConv3D(out_1,[16,16],(3,3,new_z),activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,(1,1,params["max_pool"][str(slice)+".long.2"]))
        new_z = int(constants.NUMBER_OF_IMAGE_PER_SECTION/params["max_pool"][str(slice)+".long.2"])
        out_3 = model_utils.blockConv3D(out_2,[32,32],(3,3,new_z),activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,(1,1,params["max_pool"][str(slice)+".long.3"]))
        if drop: out_3 = Dropout(params["dropout"][str(slice)+".long.1"])(out_3)
        block_3.append(out_3)
        out_4 = model_utils.blockConv3D(out_3,[8,16],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_4.append(out_4)
        out_5 = model_utils.blockConv3D(out_4,[16,32],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_5.append(out_5)
        out_6 = model_utils.blockConv3D(out_5,[32,64],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_6.append(out_6)
        out_7 = model_utils.blockConv3D(out_6,[64,128],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        if drop: out_7 = Dropout(params["dropout"][str(slice) + ".1"])(out_7)
        conv_out.append(out_7)

    if usePMs:
        layersAfterTransferLearning = []
        PMS = model_utils.getPMsList(multiInput, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)

        for pm in PMS:
            layersAfterTransferLearning.append(pm.conv_2)
            inputs.append(pm.input)
            block5_conv3 = pm.layer_dict["block5_conv3" + pm.name].output
            conv_out.append(layers.Reshape((block5_conv3.shape[1],block5_conv3.shape[2],1,block5_conv3.shape[3]))(block5_conv3))
            block4_conv3 = pm.layer_dict["block4_conv3" + pm.name].output
            block_6.append(layers.Reshape((block4_conv3.shape[1],block4_conv3.shape[2],1,block4_conv3.shape[3]))(block4_conv3))
            block3_conv3 = pm.layer_dict["block3_conv3" + pm.name].output
            block_5.append(layers.Reshape((block3_conv3.shape[1],block3_conv3.shape[2],1,block3_conv3.shape[3]))(block3_conv3))
            block2_conv2 = pm.layer_dict["block2_conv2" + pm.name].output
            block_4.append(layers.Reshape((block2_conv2.shape[1],block2_conv2.shape[2],1,block2_conv2.shape[3]))(block2_conv2))
            block1_conv2 = pm.layer_dict["block1_conv2" + pm.name].output
            block_3.append(layers.Reshape((block1_conv2.shape[1],block1_conv2.shape[2],1,block1_conv2.shape[3]))(block1_conv2))

        conc_layer = layers.Concatenate(-1)(layersAfterTransferLearning)
        transp_1 = layers.Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2), padding='same',activation=activ_func,
                                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_layer)
        conv_out.append(layers.Reshape((transp_1.shape[1],transp_1.shape[2],1,transp_1.shape[3]))(transp_1))

    # check if there is a need to add more info in the input (NIHSS, gender, ...)
    inputs, conv_out = model_utils.addMoreInfo(multiInput, inputs, conv_out, is3D=True, is4D=True)
    if len(conv_out)>1: conv_out = layers.Concatenate(-1)(conv_out)
    elif len(conv_out)==1: conv_out = conv_out[0]
    if attentiongate:
        block_6_conc = layers.Concatenate(-1)(block_6)
        attGate_1 = model_utils.attentionGateBlock(x=block_6_conc, g=conv_out, inter_shape=64, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_1 = layers.Concatenate(-1)([layers.UpSampling3D(size=size_two)(conv_out),attGate_1])
        block_5_conc = layers.Concatenate(-1)(block_5)
        up_2 = model_utils.upSamplingPlusAttention(up_1,block_5_conc,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        block_4_conc = layers.Concatenate(-1)(block_4)
        up_3 = model_utils.upSamplingPlusAttention(up_2,block_4_conc,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        block_3_conc = layers.Concatenate(-1)(block_3)
        up_4 = model_utils.upSamplingPlusAttention(up_3,block_3_conc,[8,8,8],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)

    else:
        # TODO: use this part in combination with PMs (?)
        transp_1 = Conv3DTranspose(128, kernel_size=size_two, strides=size_two, activation=activ_func,
                                   padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_out)
        if leaky: transp_1 = layers.LeakyReLU(alpha=0.33)(transp_1)

        block_6_conc = layers.Concatenate(-1)(block_6)
        up_1 = layers.Concatenate(-1)([transp_1, block_6_conc])

        up_2 = model_utils.upLayers(up_1,block_5,[64,64,64],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        up_3 = model_utils.upLayers(up_2,block_4,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        up_4 = model_utils.upLayers(up_3,block_3,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    final_conv = Conv3D(n_chann, kernel_size=(1,1,1), activation=act_name, padding='same', kernel_regularizer=l1_l2_reg,
                        kernel_initializer=kernel_init,kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_4)
    y = layers.Reshape(shape_output)(final_conv)

    model = models.Model(inputs=inputs, outputs=[y])
    return model

