import constants

from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


################################################################################
# mJ-Net model
def mJNet(X, params, drop=False, longJ=False):
    # from (30,M,N) to (1,M,N)

    # input : (30, 32, 32, 1)
    input_x = layers.Input(shape=X.shape[1:], sparse=False)
    print(K.int_shape(input_x))
    if longJ:
        conv_01 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(input_x)
        conv_01 = layers.BatchNormalization()(conv_01)
        # conv_01 : (30, 32, 32, 16)
        print(K.int_shape(conv_01))
        conv_01 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        # conv_01 : (30, 32, 32, 32)
        print(K.int_shape(conv_01))
        pool_drop_01 = layers.MaxPooling3D((2,1,1))(conv_01)
        # pool_drop_01 : (15, 32, 32, 32)
        print(K.int_shape(pool_drop_01))
        conv_02 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(pool_drop_01)
        conv_02 = layers.BatchNormalization()(conv_02)
        # conv_02 : (15, 32, 32, 16)
        print(K.int_shape(conv_02))
        conv_02 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        # conv_02 : (15, 32, 32, 32)
        print(K.int_shape(conv_02))
        pool_drop_02 = layers.MaxPooling3D((3,1,1))(conv_02)
        # pool_drop_02 : (5, 32, 16, 32)
        print(K.int_shape(pool_drop_02))
        conv_03 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(pool_drop_02)
        conv_03 = layers.BatchNormalization()(conv_03)
        # conv_03 : (5, 32, 16, 16)
        print(K.int_shape(conv_03))
        conv_03 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        # conv_03 : (5, 32, 16, 32)
        print(K.int_shape(conv_03))
        pool_drop_1 = layers.MaxPooling3D((5,1,1))(conv_03)
        # pool_drop_1 : (1, 32, 32, 32)
        print(K.int_shape(pool_drop_1))
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["0.1"])(pool_drop_1)
    else:
        conv_1 = layers.Conv3D(16, kernel_size=(constants.getN()UMBER_OF_IMAGE_PER_SECTION,3,3), activation='relu', padding='same')(input_x)
        conv_1 = layers.BatchNormalization()(conv_1)
        # conv_1 : (30, 32, 32, 16)
        print(K.int_shape(conv_1))
        pool_drop_1 = layers.AveragePooling3D((constants.getN()UMBER_OF_IMAGE_PER_SECTION,1,1))(conv_1)
        # pool_drop_1 : (1, 32, 32, 16)
        print(K.int_shape(pool_drop_1))
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (1,M,N) to (1,M/2,N/2)
    conv_2 = layers.Conv3D(32, (1,3,3), activation='relu', padding='same')(pool_drop_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    # conv_2 : (1, 32, 32, 32)
    print(K.int_shape(conv_2))
    conv_2 = layers.Conv3D(64, (1,3,3), activation='relu', padding='same')(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    # conv_2 : (1, 32, 32, 64)
    print(K.int_shape(conv_2))
    pool_drop_2 = layers.MaxPooling3D((1,2,2))(conv_2)
    # pool_drop_2 : (1, 16, 16, 64)
    print(K.int_shape(pool_drop_2))
    if drop: pool_drop_2 = layers.Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (1,M/2,N/2) to (1,M/4,N/4)
    conv_3 = layers.Conv3D(64, (1,2,2), activation='relu', padding='same')(pool_drop_2)
    conv_3 = layers.BatchNormalization()(conv_3)
    # conv_3 : (1, 16, 16, 64)
    print(K.int_shape(conv_3))
    conv_3 = layers.Conv3D(128, (1,3,3), activation='relu', padding='same')(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    # conv_3 : (1, 16, 16, 128)
    print(K.int_shape(conv_3))
    pool_drop_3 = layers.MaxPooling3D((1,2,2))(conv_3)
    # pool_drop_3 : (1, 8, 8, 128)
    print(K.int_shape(pool_drop_3))
    if drop: pool_drop_3 = layers.Dropout(params["dropout"]["3"])(pool_drop_3)

    # last convolutional layers
    conv_4 = layers.Conv3D(128, (1,2,2), activation='relu', padding='same')(pool_drop_3)
    conv_4 = layers.BatchNormalization()(conv_4)
    # conv_4 : (1, 8, 8, 128)
    print(K.int_shape(conv_4))
    conv_4 = layers.Conv3D(256, (1,3,3), activation='relu', padding='same')(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    # conv_4 : (1, 8, 8, 256)
    print(K.int_shape(conv_4))

    # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
    up_1 = layers.concatenate([layers.Conv3DTranspose(128, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same')(conv_4), conv_3], axis=1)
    # up_1 : (2, 16, 16, 128)
    print(K.int_shape(up_1))
    conv_5 = layers.Conv3D(128, (1,2,2), activation='relu', padding='same')(up_1)
    conv_5 = layers.BatchNormalization()(conv_5)
    # conv_5 : (2, 16, 16, 128)
    print(K.int_shape(conv_5))
    conv_5 = layers.Conv3D(64, (1,3,3), activation='relu', padding='same')(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    # conv_5 : (2, 16, 16, 64)
    print(K.int_shape(conv_5))
    pool_drop_4 = layers.MaxPooling3D((2,1,1))(conv_5)
    # pool_drop_4 : (1, 16, 16, 64)
    print(K.int_shape(pool_drop_4))
    if drop: pool_drop_4 = layers.Dropout(params["dropout"]["4"])(pool_drop_4)

    # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
    up_2 = layers.concatenate([layers.Conv3DTranspose(64, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same')(pool_drop_4), conv_2], axis=1)
    # up_2 : (2, 32, 32, 64)
    print(K.int_shape(up_2))
    conv_6 = layers.Conv3D(32, (1,2,2), activation='relu', padding='same')(up_2)
    conv_6 = layers.BatchNormalization()(conv_6)
    # conv_6 : (2, 32, 32, 32)
    print(K.int_shape(conv_6))
    conv_6 = layers.Conv3D(16, (1,3,3), activation='relu', padding='same')(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    # conv_6 : (2, 32, 32, 16)
    print(K.int_shape(conv_6))
    # from (2,M,N)  to (1,M,N)
    pool_drop_5 = layers.MaxPooling3D((2,1,1))(conv_6)
    # pool_drop_5 : (1, 32, 32, 16)
    print(K.int_shape(pool_drop_5))
    if drop: pool_drop_5 = layers.Dropout(params["dropout"]["5"])(pool_drop_5)

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = layers.Conv3D(1, (1,1,1), activation="sigmoid", padding='same')(pool_drop_5)
    # conv_7 : (1, 32, 32, 16)
    print(K.int_shape(conv_7))
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
