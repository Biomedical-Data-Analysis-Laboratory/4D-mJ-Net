import constants
from Utils import general_utils, spatial_pyramid

from tensorflow.keras import layers, models, regularizers, initializers
import tensorflow.keras.backend as K


################################################################################
# mJ-Net model
def mJNet(X, params, to_categ, drop=False, longJ=False, v2=False):
    # from (30,M,N) to (1,M,N)

    size_two = (1,2,2)
    activ_func = 'relu'
    l1_l2_reg = None
    channels = [16,32,16,32,16,32,16,32,64,64,128,128,256,-1,-1,-1,-1,128,128,64,64,32,16]
    input_shape = X.shape[1:]
    kernel_init = "glorot_uniform" # Xavier uniform initializer.

    if v2: # version 2
        # size_two = (2,2,1)
        activ_func = None
        l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
        # Hu initializer
        kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)

        # channels = [16,32,32,64,64,128,128,32,64,128,256,512,1024,512,1024,512,1024,-1,512,256,-1,128,64]
        channels = [16,32,32,64,64,128,128,16,32,32,64,64,128,128,128,256,128,-1,128,64,-1,64,32]
        channels = [int(ch/2) for ch in channels] # implemented due to memory issues

        # input_shape = (None,constants.getM(),constants.getN(),1)
        ## TODO: input_shape = (constants.NUMBER_OF_IMAGE_PER_SECTION,None,None,1)

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x) # (None, 30, M, N, 1)

    if longJ:
        conv_01 = layers.Conv3D(channels[0], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(input_x)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01) # (None, 30, M, N, 16)
        conv_01 = layers.Conv3D(channels[1], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_01)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01) # (None, 30, M, N, 32)

        pool_drop_01 = layers.MaxPooling3D((params["max_pool"]["long.1"],1,1))(conv_01)
        general_utils.print_int_shape(pool_drop_01) # (None, 15, M, N, 32)
        conv_02 = layers.Conv3D(channels[2], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_01)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02) # (None, 15, M, N, 32)
        conv_02 = layers.Conv3D(channels[3], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_02)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02) # (None, 15, M, N, 64)

        pool_drop_02 = layers.MaxPooling3D((params["max_pool"]["long.2"],1,1))(conv_02)
        general_utils.print_int_shape(pool_drop_02) # (None, 5, M, N, 64)
        conv_03 = layers.Conv3D(channels[4], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_02)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03) # (None, 5, M, N, 64)
        conv_03 = layers.Conv3D(channels[5], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_03)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03) # (None, 5, M, N, 128)
        pool_drop_1 = layers.MaxPooling3D((params["max_pool"]["long.3"],1,1))(conv_03)
        general_utils.print_int_shape(pool_drop_1) # (None, 1, M, N, 128)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["long.1"])(pool_drop_1)
    else:
        # conv_1 = layers.Conv3D(channels[6], kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(input_x)
        conv_1 = layers.Conv3D(channels[6], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(input_x)
        if v2: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
        conv_1 = layers.BatchNormalization()(conv_1)
        general_utils.print_int_shape(conv_1) # (None, 30, M, N, 128)
        pool_drop_1 = layers.AveragePooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(conv_1)
        # pool_drop_1 = spatial_pyramid.SPP3D([1,2,4], input_shape=(channels[6],None,None,None))(conv_1)
        general_utils.print_int_shape(pool_drop_1) # (None, 1, M, N, 128)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (1,M,N) to (1,M/2,N/2)
    conv_2 = layers.Conv3D(channels[7], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_1)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2) # (None, 1, M, N, 32)
    conv_2 = layers.Conv3D(channels[8], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_2)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2) # (None, 1, M, N, 64)
    pool_drop_2 = layers.MaxPooling3D(size_two)(conv_2)
    general_utils.print_int_shape(pool_drop_2) # (None, 1, M/2, N/2, 64)
    if drop: pool_drop_2 = layers.Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (1,M/2,N/2) to (1,M/4,N/4)
    conv_3 = layers.Conv3D(channels[9], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_2)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3) # (None, 1, M/2, N/2, 128)
    conv_3 = layers.Conv3D(channels[10], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_3)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3) # (None, 1, M/2, N/2, 256)
    pool_drop_3 = layers.MaxPooling3D(size_two)(conv_3)
    general_utils.print_int_shape(pool_drop_3) # (None, 1, M/4, N/4, 256)
    if drop: pool_drop_3 = layers.Dropout(params["dropout"]["3"])(pool_drop_3)

    # from (1,M/4,N/4) to (1,M/8,N/8)
    conv_4 = layers.Conv3D(channels[11], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_3)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4) # (None, 1, M/4, N/4, 512)
    conv_4 = layers.Conv3D(channels[12], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_4)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4) # (None, 1, M/4, N/4, 1024)

    if v2:
        pool_drop_3_1 = layers.MaxPooling3D(size_two)(conv_4)
        general_utils.print_int_shape(pool_drop_3_1) # (None, 1, M/8, N/8, 1024)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.1"])(pool_drop_3_1)
        conv_4_1 = layers.Conv3D(channels[13], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_3_1)
        conv_4_1 = layers.LeakyReLU(alpha=0.33)(conv_4_1)
        conv_4_1 = layers.BatchNormalization()(conv_4_1)
        general_utils.print_int_shape(conv_4_1) # (None, 1, M/8, N/8, 512)
        if drop: pool_drop_4_1 = layers.Dropout(params["dropout"]["3.2"])(conv_4_1)
        conv_5_1 = layers.Conv3D(channels[14], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_4_1)
        conv_5_1 = layers.LeakyReLU(alpha=0.33)(conv_5_1)
        conv_5_1 = layers.BatchNormalization()(conv_5_1)
        general_utils.print_int_shape(conv_5_1) # (None, 1, M/8, N/8, 1024)
        add_1 = layers.add([pool_drop_3_1, conv_5_1])
        general_utils.print_int_shape(add_1) # (None, 1, M/8, N/8, 1024)
        up_01 = layers.UpSampling3D(size=size_two)(add_1)
        general_utils.print_int_shape(up_01) # (None, 1, M/4, N/4, 1024)

        conc_1 = layers.concatenate([up_01, conv_4], axis=-1)
        general_utils.print_int_shape(conc_1) # (None, 1, M/4, N/4, 1024)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.3"])(pool_drop_3_1)
        conv_6_1 = layers.Conv3D(channels[15], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conc_1)
        conv_6_1 = layers.LeakyReLU(alpha=0.33)(conv_6_1)
        conv_6_1 = layers.BatchNormalization()(conv_6_1)
        general_utils.print_int_shape(conv_6_1) # (None, 1, M/4, N/4, 512)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.4"])(pool_drop_3_1)
        conv_7_1 = layers.Conv3D(channels[16], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_6_1)
        conv_7_1 = layers.LeakyReLU(alpha=0.33)(conv_7_1)
        conv_7_1 = layers.BatchNormalization()(conv_7_1)
        general_utils.print_int_shape(conv_7_1) # (None, 1, M/4, N/4, 1024)
        add_2 = layers.add([conv_4, conv_7_1])
        general_utils.print_int_shape(add_2) # (None, 1, M/4, N/4, 1024)
        up_02 = layers.UpSampling3D(size=size_two)(add_2)
        general_utils.print_int_shape(up_02) # (None, 1, M/2, N/2, 1024)

        addconv_3 = layers.concatenate([conv_3, conv_3])
        while K.int_shape(addconv_3)[-1] !=  K.int_shape(up_02)[-1]:
            addconv_3 = layers.concatenate([addconv_3, addconv_3])
        up_1 = layers.concatenate([up_02, addconv_3])
    else:
        # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
        up_1 = layers.concatenate([layers.Conv3DTranspose(channels[17], kernel_size=size_two, strides=size_two, activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_4), conv_3], axis=1)

    general_utils.print_int_shape(up_1) # (None, 1, M/2, N/2, 1024)
    conv_5 = layers.Conv3D(channels[18], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_1)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5) # (None, 1, M/2, N/2, 512)
    conv_5 = layers.Conv3D(channels[19], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_5)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5) # (None, 1, M/2, N/2, 256)

    if v2:
        addconv_5 = layers.concatenate([conv_5, conv_5])
        while K.int_shape(addconv_5)[-1] !=  K.int_shape(up_1)[-1]:
            addconv_5 = layers.concatenate([addconv_5, addconv_5])
        add_3 = layers.add([up_1, addconv_5])
        general_utils.print_int_shape(add_3) # (None, 1, M/2, N/4, 1024)
        up_03 = layers.UpSampling3D(size=size_two)(add_3)
        general_utils.print_int_shape(up_02) # (None, 1, M, N, 1024)

        addconv_2 = layers.concatenate([conv_2, conv_2])
        while K.int_shape(addconv_2)[-1] !=  K.int_shape(up_03)[-1]:
            addconv_2 = layers.concatenate([addconv_2, addconv_2])
        up_2 = layers.concatenate([up_03, addconv_2])
    else:
        pool_drop_4 = layers.MaxPooling3D((2,1,1))(conv_5)
        general_utils.print_int_shape(pool_drop_4) # (None, 1, M/2, N/2, 512)
        if drop: pool_drop_4 = layers.Dropout(params["dropout"]["4"])(pool_drop_4)
        # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
        up_2 = layers.concatenate([layers.Conv3DTranspose(channels[20], kernel_size=size_two, strides=size_two, activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_4), conv_2], axis=1)

    general_utils.print_int_shape(up_2) # (None, X, M, N, 1024)
    conv_6 = layers.Conv3D(channels[21], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_2)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6) # (None, X, M, N, 128)
    conv_6 = layers.Conv3D(channels[22], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_6)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    pool_drop_5 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(pool_drop_5) # (None, X, M, N, 64)

    if not v2:
        # from (2,M,N)  to (1,M,N)
        pool_drop_5 = layers.MaxPooling3D((2,1,1))(pool_drop_5)
        general_utils.print_int_shape(pool_drop_5) # (None, 1, M, N, 16)
        if drop: pool_drop_5 = layers.Dropout(params["dropout"]["5"])(pool_drop_5)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(),constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(),constants.getN(),n_chann)

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = layers.Conv3D(n_chann, (1,1,1), activation=act_name, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_5)
    general_utils.print_int_shape(conv_7) # (None, 1, M, N, 1)
    y = layers.Reshape(shape_output)(conv_7)
    general_utils.print_int_shape(y) # (None, M, N)
    model = models.Model(inputs=input_x, outputs=y)

    return model
