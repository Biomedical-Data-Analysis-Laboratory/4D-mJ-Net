import constants

from tensorflow.keras import layers, models, regularizers, initializers
import tensorflow.keras.backend as K


################################################################################
# mJ-Net model
def mJNet(X, params, drop=False, longJ=False, v2=False):
    # from (30,M,N) to (1,M,N)

    activ_func = 'relu'
    l1_l2_reg = None
    channels = [16,32,16,32,16,32,16,32,64,64,128,128,256,-1,-1,-1,-1,128,128,64,32,16]

    if v2: # version 2
        activ_func = None
        l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
        channels = [16,32,32,64,64,128,128,32,64,128,256,512,1024,512,1024,512,1024,-1,512,256,-1,128,64]

    input_x = layers.Input(shape=X.shape[1:], sparse=False)
    print(K.int_shape(input_x)) # (None, 30, M, N, 1)
    if longJ:
        conv_01 = layers.Conv3D(channels[0], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(input_x)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        print(K.int_shape(conv_01)) # (None, 30, M, N, 16)
        conv_01 = layers.Conv3D(channels[1], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_01)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        print(K.int_shape(conv_01)) # (None, 30, M, N, 32)

        pool_drop_01 = layers.MaxPooling3D((2,1,1))(conv_01)
        print(K.int_shape(pool_drop_01)) # (None, 15, M, N, 32)
        conv_02 = layers.Conv3D(channels[2], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_01)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        print(K.int_shape(conv_02)) # (None, 15, M, N, 32)
        conv_02 = layers.Conv3D(channels[3], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_02)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        print(K.int_shape(conv_02)) # (None, 15, M, N, 64)

        pool_drop_02 = layers.MaxPooling3D((3,1,1))(conv_02)
        print(K.int_shape(pool_drop_02)) # (None, 5, M, N, 64)
        conv_03 = layers.Conv3D(channels[4], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_02)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        print(K.int_shape(conv_03)) # (None, 5, M, N, 64)
        conv_03 = layers.Conv3D(channels[5], kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_03)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        print(K.int_shape(conv_03)) # (None, 5, M, N, 128)
        pool_drop_1 = layers.MaxPooling3D((5,1,1))(conv_03)
        print(K.int_shape(pool_drop_1)) # (None, 1, M, N, 128)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["0.1"])(pool_drop_1)
    else:
        conv_1 = layers.Conv3D(channels[6], kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(input_x)
        conv_1 = layers.BatchNormalization()(conv_1)
        print(K.int_shape(conv_1)) # (None, 30, M, N, 16)
        pool_drop_1 = layers.AveragePooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(conv_1)
        print(K.int_shape(pool_drop_1)) # (None, 1, M, N, 128)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (1,M,N) to (1,M/2,N/2)
    conv_2 = layers.Conv3D(channels[7], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_1)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    print(K.int_shape(conv_2)) # (None, 1, M, N, 32)
    conv_2 = layers.Conv3D(channels[8], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_2)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    print(K.int_shape(conv_2)) # (None, 1, M, N, 64)
    pool_drop_2 = layers.MaxPooling3D((1,2,2))(conv_2)
    print(K.int_shape(pool_drop_2)) # (None, 1, M/2, N/2, 64)
    if drop: pool_drop_2 = layers.Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (1,M/2,N/2) to (1,M/4,N/4)
    conv_3 = layers.Conv3D(channels[9], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_2)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    print(K.int_shape(conv_3)) # (None, 1, M/2, N/2, 128)
    conv_3 = layers.Conv3D(channels[10], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_3)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    print(K.int_shape(conv_3)) # (None, 1, M/2, N/2, 256)
    pool_drop_3 = layers.MaxPooling3D((1,2,2))(conv_3)
    print(K.int_shape(pool_drop_3)) # (None, 1, M/4, N/4, 256)
    if drop: pool_drop_3 = layers.Dropout(params["dropout"]["3"])(pool_drop_3)

    # from (1,M/4,N/4) to (1,M/8,N/8)
    conv_4 = layers.Conv3D(channels[11], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_3)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    print(K.int_shape(conv_4)) # (None, 1, M/4, N/4, 512)
    conv_4 = layers.Conv3D(channels[12], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_4)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    print(K.int_shape(conv_4)) # (None, 1, M/4, N/4, 1024)

    if v2:
        pool_drop_3_1 = layers.MaxPooling3D((1,2,2))(conv_4)
        print(K.int_shape(pool_drop_3_1)) # (None, 1, M/8, N/8, 1024)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.1"])(pool_drop_3_1)
        conv_4_1 = layers.Conv3D(channels[13], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_3_1)
        conv_4_1 = layers.LeakyReLU(alpha=0.33)(conv_4_1)
        conv_4_1 = layers.BatchNormalization()(conv_4_1)
        print(K.int_shape(conv_4_1)) # (None, 1, M/8, N/8, 512)
        if drop: pool_drop_4_1 = layers.Dropout(params["dropout"]["3.2"])(conv_4_1)
        conv_5_1 = layers.Conv3D(channels[14], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_4_1)
        conv_5_1 = layers.LeakyReLU(alpha=0.33)(conv_5_1)
        conv_5_1 = layers.BatchNormalization()(conv_5_1)
        print(K.int_shape(conv_5_1)) # (None, 1, M/8, N/8, 1024)
        add_1 = layers.add([pool_drop_3_1, conv_5_1])
        print(K.int_shape(add_1)) # (None, 1, M/8, N/8, 1024)
        up_01 = layers.UpSampling3D(size=(1,2,2))(add_1)
        print(K.int_shape(up_01)) # (None, 1, M/4, N/4, 1024)

        conc_1 = layers.concatenate([up_01, conv_4], axis=-1)
        print(K.int_shape(conc_1)) # (None, 1, M/4, N/4, 1024)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.3"])(pool_drop_3_1)
        conv_6_1 = layers.Conv3D(channels[15], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conc_1)
        conv_6_1 = layers.LeakyReLU(alpha=0.33)(conv_6_1)
        conv_6_1 = layers.BatchNormalization()(conv_6_1)
        print(K.int_shape(conv_6_1)) # (None, 1, M/4, N/4, 512)
        if drop: pool_drop_3_1 = layers.Dropout(params["dropout"]["3.4"])(pool_drop_3_1)
        conv_7_1 = layers.Conv3D(channels[16], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_6_1)
        conv_7_1 = layers.LeakyReLU(alpha=0.33)(conv_7_1)
        conv_7_1 = layers.BatchNormalization()(conv_7_1)
        print(K.int_shape(conv_7_1)) # (None, 1, M/4, N/4, 1024)
        add_2 = layers.add([conv_4, conv_7_1])
        print(K.int_shape(add_2)) # (None, 1, M/4, N/4, 1024)
        up_02 = layers.UpSampling3D(size=(1,2,2))(add_2)
        print(K.int_shape(up_02)) # (None, 1, M/2, N/2, 1024)

        addconv_3 = layers.concatenate([conv_3, conv_3])
        while K.int_shape(addconv_3)[-1] !=  K.int_shape(up_02)[-1]:
            addconv_3 = layers.concatenate([addconv_3, addconv_3])
        up_1 = layers.concatenate([up_02, addconv_3])
    else:
        # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
        up_1 = layers.concatenate([layers.Conv3DTranspose(channels[17], kernel_size=(1,2,2), strides=(1,2,2), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_4), conv_3], axis=1)

    print(K.int_shape(up_1)) # (None, 1, M/2, N/2, 1024)
    conv_5 = layers.Conv3D(channels[18], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(up_1)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    print(K.int_shape(conv_5)) # (None, 1, M/2, N/2, 512)
    conv_5 = layers.Conv3D(channels[19], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_5)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    print(K.int_shape(conv_5)) # (None, 1, M/2, N/2, 256)

    if v2:
        addconv_5 = layers.concatenate([conv_5, conv_5])
        while K.int_shape(addconv_5)[-1] !=  K.int_shape(up_1)[-1]:
            addconv_5 = layers.concatenate([addconv_5, addconv_5])
        add_3 = layers.add([up_1, addconv_5])
        print(K.int_shape(add_3)) # (None, 1, M/2, N/4, 1024)
        up_03 = layers.UpSampling3D(size=(1,2,2))(add_3)
        print(K.int_shape(up_02)) # (None, 1, M, N, 1024)

        addconv_2 = layers.concatenate([conv_2, conv_2])
        while K.int_shape(addconv_2)[-1] !=  K.int_shape(up_03)[-1]:
            addconv_2 = layers.concatenate([addconv_2, addconv_2])
        up_2 = layers.concatenate([up_03, addconv_2])
    else:
        pool_drop_4 = layers.MaxPooling3D((2,1,1))(conv_5)
        print(K.int_shape(pool_drop_4)) # (None, 1, M/2, N/2, 512)
        if drop: pool_drop_4 = layers.Dropout(params["dropout"]["4"])(pool_drop_4)
        # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
        up_2 = layers.concatenate([layers.Conv3DTranspose(channels[20], kernel_size=(1,2,2), strides=(1,2,2), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_4), conv_2], axis=1)

    print(K.int_shape(up_2)) # (None, X, M, N, 1024)
    conv_6 = layers.Conv3D(channels[21], (3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(up_2)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    print(K.int_shape(conv_6)) # (None, X, M, N, 128)
    conv_6 = layers.Conv3D(channels[22], (1,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(conv_6)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    pool_drop_5 = layers.BatchNormalization()(conv_6)
    print(K.int_shape(pool_drop_5)) # (None, X, M, N, 64)

    if not v2:
        # from (2,M,N)  to (1,M,N)
        pool_drop_5 = layers.MaxPooling3D((2,1,1))(pool_drop_5)
        print(K.int_shape(pool_drop_5)) # (None, 1, M, N, 16)
        if drop: pool_drop_5 = layers.Dropout(params["dropout"]["5"])(pool_drop_5)

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = layers.Conv3D(1, (1,1,1), activation="sigmoid", padding='same', kernel_regularizer=l1_l2_reg)(pool_drop_5)
    print(K.int_shape(conv_7)) # (None, 1, M, N, 1)
    y = layers.Reshape((constants.getM(),constants.getN()))(conv_7)
    print(K.int_shape(y))
    model = models.Model(inputs=input_x, outputs=y)
    return model

################################################################################
# Function to call the mJ-net with dropout
def mJNet_Drop(X, params):
    return mJNet(X, params, drop=True)

################################################################################
# Function to call the mJ-net with dropout and a long "J"
def mJNet_LongJ_Drop(X, params):
    return mJNet(X, params, drop=True, longJ=True)

################################################################################
# Function to call the mJ-net with a long "J"
def mJNet_LongJ(X, params):
    return mJNet(X, params, longJ=True)

################################################################################
# mJ-Net model version 2
def mJNet_v2(X, params):
    return mJNet(X, params, drop=True, longJ=True, v2=True)

################################################################################
# Model from Van De Leemput (https://doi.org/10.1109/ACCESS.2019.2910348)
def van_De_Leemput(X, params):
    l1_l2_reg = None # regularizers.l1_l2(l1=1e-6, l2=1e-5)
    # Hu initializer = [0, sqrt(9/5*fan_in)]
    hu_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)

    input_x = layers.Input(shape=X.shape[1:], sparse=False)
    print(K.int_shape(input_x)) # (None, 30, 16, 16, 1)
    conv_1 = layers.Conv3D(32, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(input_x)
    conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)
    print(K.int_shape(conv_1)) # (None, 30, 16, 16, 32)
    conv_2 = layers.Conv3D(64, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_1)
    conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    print(K.int_shape(conv_2)) # (None, 30, 16, 16, 64)
    add_1 = layers.add([input_x, conv_2])
    print(K.int_shape(add_1))  # (None, 30, 16, 16, 64)

    pool_1 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,2,2))(add_1)
    print(K.int_shape(pool_1)) # (None, 1, 8, 8, 64)
    conv_3 = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(pool_1)
    conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    print(K.int_shape(conv_3)) # (None, 1, 8, 8, 64)
    conv_4 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_3)
    conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    print(K.int_shape(conv_4)) # (None, 1, 8, 8, 128)
    addpool_1 = layers.concatenate([pool_1, pool_1])
    print(K.int_shape(addpool_1)) # (None, 1, 8, 8, 128)
    add_2 = layers.add([addpool_1, conv_4])
    print(K.int_shape(add_2)) # (None, 1, 8, 8, 128)

    pool_2 = layers.MaxPooling3D((1,2,2))(add_2)
    print(K.int_shape(pool_2)) # (None, 1, 4, 4, 128)
    conv_5 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(pool_2)
    conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    print(K.int_shape(conv_5)) # (None, 1, 4, 4, 128)
    conv_6 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_5)
    conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    print(K.int_shape(conv_6)) # (None, 1, 4, 4, 256)
    addpool_2 = layers.concatenate([pool_2, pool_2])
    print(K.int_shape(addpool_2)) # (None, 1, 4, 4, 256)
    add_3 = layers.add([addpool_2, conv_6])
    print(K.int_shape(add_3)) # (None, 1, 4, 4, 256)

    pool_3 = layers.MaxPooling3D((1,2,2))(add_3)
    print(K.int_shape(pool_3)) # (None, 1, 2, 2, 256)
    pool_3 = layers.Dropout(params["dropout"]["1"])(pool_3)
    conv_7 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(pool_3)
    conv_7 = layers.LeakyReLU(alpha=0.33)(conv_7)
    conv_7 = layers.BatchNormalization()(conv_7)
    conv_7 = layers.Dropout(params["dropout"]["2"])(conv_7)
    print(K.int_shape(conv_7)) # (None, 1, 2, 2, 256)
    conv_8 = layers.Conv3D(512, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_7)
    conv_8 = layers.LeakyReLU(alpha=0.33)(conv_8)
    conv_8 = layers.BatchNormalization()(conv_8)
    print(K.int_shape(conv_8)) # (None, 1, 2, 2, 512)
    addpool_3 = layers.concatenate([pool_3, pool_3])
    print(K.int_shape(addpool_3)) # (None, 1, 2, 2, 512)
    add_4 = layers.add([addpool_3, conv_8])
    print(K.int_shape(add_4)) # (None, 1, 2, 2, 512)
    up_1 = layers.UpSampling3D(size=(1,2,2))(add_4)
    print(K.int_shape(up_1)) # (None, 1, 4, 4, 512)

    addadd_3 = layers.concatenate([add_3, add_3], axis=-1)
    print(K.int_shape(addadd_3)) # (None, 1, 4, 4, 512)
    conc_1 = layers.concatenate([up_1, addadd_3], axis=-1)
    print(K.int_shape(conc_1)) # (None, 1, 4, 4, 1024)
    conv_9 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conc_1)
    conv_9 = layers.LeakyReLU(alpha=0.33)(conv_9)
    conv_9 = layers.BatchNormalization()(conv_9)
    print(K.int_shape(conv_9)) # (None, 1, 4, 4, 256)
    conv_10 = layers.Conv3D(512, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_9)
    conv_10 = layers.LeakyReLU(alpha=0.33)(conv_10)
    conv_10 = layers.BatchNormalization()(conv_10)
    print(K.int_shape(conv_10)) # (None, 1, 4, 4, 512)
    addconv_10 = layers.concatenate([conv_10, conv_10])
    print(K.int_shape(addconv_10)) # (None, 1, 4, 4, 1024)
    add_5 = layers.add([conc_1, addconv_10])
    print(K.int_shape(add_5)) # (None, 1, 4, 4, 1024)
    up_2 = layers.UpSampling3D(size=(1,2,2))(add_5)
    print(K.int_shape(up_2)) # (None, 1, 8, 8, 1024)

    addadd_2 = layers.concatenate([add_2, add_2])
    while K.int_shape(addadd_2)[-1] !=  K.int_shape(up_2)[-1]:
        addadd_2 = layers.concatenate([addadd_2, addadd_2])
    print(K.int_shape(addadd_2)) # (None, 1, 8, 8, 1024)
    conc_2 = layers.concatenate([up_2, addadd_2], axis=-1)
    print(K.int_shape(conc_2)) # (None, 1, 8, 8, 2048)
    conv_11 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conc_2)
    conv_11 = layers.LeakyReLU(alpha=0.33)(conv_11)
    conv_11 = layers.BatchNormalization()(conv_11)
    print(K.int_shape(conv_11)) # (None, 1, 8, 8, 128)
    conv_12 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_11)
    conv_12 = layers.LeakyReLU(alpha=0.33)(conv_12)
    conv_12 = layers.BatchNormalization()(conv_12)
    print(K.int_shape(conv_12)) # (None, 1, 8, 8, 128)
    addconv_12 = layers.concatenate([conv_12, conv_12])
    while K.int_shape(addconv_12)[-1] !=  K.int_shape(conc_2)[-1]:
        addconv_12 = layers.concatenate([addconv_12, addconv_12])
    print(K.int_shape(addconv_12)) # (None, 1, 8, 8, 2048)
    add_6 = layers.add([conc_2, addconv_12])
    print(K.int_shape(add_6)) # (None, 1, 8, 8, 2048)
    up_3 = layers.UpSampling3D(size=(1,2,2))(add_6)
    print(K.int_shape(up_3)) # (None, 1, 16, 16, 2048)

    addpool_0 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(add_1)
    addadd_1 = layers.concatenate([addpool_0,addpool_0])
    while K.int_shape(addadd_1)[-1] !=  K.int_shape(up_3)[-1]:
        addadd_1 = layers.concatenate([addadd_1, addadd_1])
    print(K.int_shape(addadd_1)) # (None, 1, 16, 16, 2048)
    conc_3 = layers.concatenate([up_3, addadd_1], axis=-1)
    print(K.int_shape(conc_3)) # (None, 1, 16, 16, 2048)
    conv_13 = layers.Conv3D(64, kernel_size=(1,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conc_3)
    conv_13 = layers.LeakyReLU(alpha=0.33)(conv_13)
    conv_13 = layers.BatchNormalization()(conv_13)
    print(K.int_shape(conv_13)) # (None, 1, 16, 16, 64)
    conv_14 = layers.Conv3D(64, kernel_size=(1,3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(conv_13)
    conv_14 = layers.LeakyReLU(alpha=0.33)(conv_14)
    conv_14 = layers.BatchNormalization()(conv_14)
    print(K.int_shape(conv_14)) # (None, 1, 16, 16, 64)
    addconv_14 = layers.concatenate([conv_14,conv_14])
    while K.int_shape(addconv_14)[-1] !=  K.int_shape(conc_3)[-1]:
        addconv_14 = layers.concatenate([addconv_14, addconv_14])
    print(K.int_shape(addconv_14)) # (None, 1, 16, 16, 2048)
    add_7 = layers.add([conc_3, addconv_14])
    print(K.int_shape(add_7)) # (None, 1, 16, 16, 2048)

    conv_15 = layers.Conv3D(1, (1,1,1), activation="sigmoid", padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(add_7)
    print(K.int_shape(conv_15)) # (None, 1, 16, 16, 4)
    y = layers.Reshape((constants.getM(),constants.getN()))(conv_15)
    print(K.int_shape(y))
    model = models.Model(inputs=input_x, outputs=y)
    return model

################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(X, params):
    # Hu initializer = [0, sqrt(2/fan_in)]
    hu_init = initializers.he_normal(seed=None)

    input_x = layers.Input(shape=X.shape[1:], sparse=False)
    print(K.int_shape(input_x)) # (None, 30, 16, 16, 1)
    conv_1 = layers.Conv3D(64, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(input_x)
    print(K.int_shape(conv_1)) # (None, 30, 16, 16, 64)
    conv_2 = layers.Conv3D(64, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_1)
    print(K.int_shape(conv_2)) # (None, 30, 16, 16, 64)

    pool_1 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,2,2))(conv_2)
    print(K.int_shape(pool_1)) # (None, 1, 8, 8, 64)
    conv_3 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(pool_1)
    print(K.int_shape(conv_3)) # (None, 1, 8, 8, 128)
    conv_4= layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_3)
    print(K.int_shape(conv_4)) # (None, 1, 8, 8, 128)

    pool_2 = layers.MaxPooling3D((1,2,2))(conv_4)
    print(K.int_shape(pool_2)) # (None, 1, 4, 4, 128)
    conv_5 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(pool_2)
    print(K.int_shape(conv_5)) # (None, 1, 4, 4, 256)
    conv_6= layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_5)
    print(K.int_shape(conv_6)) # (None, 1, 4, 4, 256)
    pool_3 = layers.MaxPooling3D((1,2,2))(conv_6)
    print(K.int_shape(pool_3)) # (None, 1, 2, 2, 256)

    conv_7 = layers.Conv3D(512, kernel_size=(1,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(pool_3)
    print(K.int_shape(conv_7)) # (None, 1, 2, 2, 512)
    conv_8= layers.Conv3D(512, kernel_size=(1,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_7)
    print(K.int_shape(conv_8)) # (None, 1, 2, 2, 512)
    pool_4 = layers.MaxPooling3D((1,2,2))(conv_8)
    print(K.int_shape(pool_4)) # (None, 1, 1, 1, 512)

    conv_9 = layers.Conv3D(1024, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(pool_4)
    print(K.int_shape(conv_9)) # (None, 1, 1, 1, 1024)
    conv_10 = layers.Conv3D(1024, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_9)
    print(K.int_shape(conv_10)) # (None, 1, 1, 1, 1024)

    up_1 = layers.Conv3DTranspose(512, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_10)
    # up_1 = layers.UpSampling3D(size=(1,2,2))(conv_10)
    print(K.int_shape(up_1)) # (None, 1, 2, 2, 1024)
    # addconv_8 = layers.concatenate([conv_8, conv_8])
    # print(K.int_shape(addconv_8)) # (None, 1, 2, 2, 1024)
    # conc_1 = layers.concatenate([up_1, addconv_8], axis=-1)
    conc_1 = layers.concatenate([up_1, conv_8], axis=-1)
    print(K.int_shape(conc_1)) # (None, 1, 2, 2, 1024)
    conv_11 = layers.Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_1)
    print(K.int_shape(conv_11)) # (None, 1, 2, 2, 512)
    conv_12 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_11)
    print(K.int_shape(conv_12)) # (None, 1, 2, 2, 256)

    up_2 = layers.Conv3DTranspose(256, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_12)
    # up_2 = layers.UpSampling3D(size=(1,2,2))(conv_12)
    print(K.int_shape(up_1)) # (None, 1, 4, 4, 256)
    conc_2 = layers.concatenate([up_2, conv_6], axis=-1)
    print(K.int_shape(conc_2)) # (None, 1, 4, 4, 256)
    conv_13 = layers.Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_2)
    print(K.int_shape(conv_13)) # (None, 1, 4, 4, 512)
    conv_14 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_13)
    print(K.int_shape(conv_14)) # (None, 1, 4, 4, 256)

    up_3 = layers.Conv3DTranspose(128, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_14)
    # up_3 = layers.UpSampling3D(size=(1,2,2))(conv_14)
    print(K.int_shape(up_3)) # (None, 1, 8, 8, 256)
    # addconv_4 = layers.concatenate([conv_4, conv_4])
    # print(K.int_shape(addconv_4)) # (None, 1, 8, 8, 256)
    # conc_3 = layers.concatenate([up_3, addconv_4], axis=-1)
    conc_3 = layers.concatenate([up_3, conv_4], axis=-1)
    print(K.int_shape(conc_3)) # (None, 1, 8, 8, 256)
    conv_15 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_3)
    print(K.int_shape(conv_15)) # (None, 1, 8, 8, 256)
    conv_16 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_15)
    print(K.int_shape(conv_16)) # (None, 1, 8, 8, 128)

    up_4 = layers.Conv3DTranspose(64, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_16)
    # up_4 = layers.UpSampling3D(size=(1,2,2))(conv_16)
    print(K.int_shape(up_4)) # (None, 1, 16, 16, 128)
    convpool_2 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(conv_2)
    # convpool_2 = layers.concatenate([convpool_2,convpool_2])
    conc_4 = layers.concatenate([up_4, convpool_2], axis=-1)
    print(K.int_shape(conc_4)) # (None, 1, 16, 16, 128)
    conv_17 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_4)
    print(K.int_shape(conv_17)) # (None, 1, 16, 16, 128)
    conv_18 = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_17)
    print(K.int_shape(conv_18)) # (None, 1, 16, 16, 64)
    conv_19 = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_18)
    print(K.int_shape(conv_19)) # (None, 1, 16, 16, 64)

    conv_20 = layers.Conv3D(1, (1,1,1), activation="sigmoid", padding='same', kernel_initializer=hu_init)(conv_19)
    print(K.int_shape(conv_20)) # (None, 1, 16, 16, 1)
    y = layers.Reshape((constants.getM(),constants.getN()))(conv_20)
    print(K.int_shape(y))
    model = models.Model(inputs=input_x, outputs=y)
    return model
