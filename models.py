import constants

from tensorflow.keras import layers, models

################################################################################
# mJ-Net model
def mJNet(X, params, drop=False, longJ=False):
    # from (M,N,30) to (M,N,1)
    input_x = layers.Input(shape=X.shape[1:], sparse=False)
    if longJ:
        conv_01 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(input_x)
        conv_01 = layers.BatchNormalization()(conv_01)
        conv_01 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        pool_drop_01 = layers.MaxPooling3D((1,1,2))(conv_01)
        conv_02 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(pool_drop_01)
        conv_02 = layers.BatchNormalization()(conv_02)
        conv_02 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        pool_drop_02 = layers.MaxPooling3D((1,1,3))(conv_02)
        conv_03 = layers.Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same')(pool_drop_02)
        conv_03 = layers.BatchNormalization()(conv_03)
        conv_03 = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same')(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        pool_drop_1 = layers.MaxPooling3D((1,1,5))(conv_03)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["0.1"])(pool_drop_1)
    else:
        conv_1 = layers.Conv3D(16, kernel_size=(3,3,constants.NUMBER_OF_IMAGE_PER_SECTION), activation='relu', padding='same')(input_x)
        conv_1 = layers.BatchNormalization()(conv_1)
        pool_drop_1 = layers.AveragePooling3D((1,1,constants.NUMBER_OF_IMAGE_PER_SECTION))(conv_1)
        if drop: pool_drop_1 = layers.Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (M,N,1) to (M/2,N/2,1)
    conv_2 = layers.Conv3D(32, (3,3,1), activation='relu', padding='same')(pool_drop_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    conv_2 = layers.Conv3D(64, (3,3,1), activation='relu', padding='same')(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    pool_drop_2 = layers.MaxPooling3D((2,2,1))(conv_2)
    if drop: pool_drop_2 = layers.Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (M/2,N/2,1) to (M/4,N/4,1)
    conv_3 = layers.Conv3D(64, (3,3,1), activation='relu', padding='same')(pool_drop_2)
    conv_3 = layers.BatchNormalization()(conv_3)
    conv_3 = layers.Conv3D(128, (3,3,1), activation='relu', padding='same')(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    pool_drop_3 = layers.MaxPooling3D((2,2,1))(conv_3)
    if drop: pool_drop_3 = layers.Dropout(params["dropout"]["3"])(pool_drop_3)

    # last convolutional layers
    conv_4 = layers.Conv3D(128, (3,3,1), activation='relu', padding='same')(pool_drop_3)
    conv_4 = layers.BatchNormalization()(conv_4)
    conv_4 = layers.Conv3D(256, (3,3,1), activation='relu', padding='same')(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)

    # first UP-convolutional layer: from (M/4,N/4,1) to (M/2,N/2,2)
    up_1 = layers.concatenate([layers.Conv3DTranspose(128, kernel_size=(2,2,1), strides=(2,2,1), activation='relu', padding='same')(conv_4), conv_3], axis=3)
    conv_5 = layers.Conv3D(128, (3,3,1), activation='relu', padding='same')(up_1)
    conv_5 = layers.BatchNormalization()(conv_5)
    conv_5 = layers.Conv3D(64, (3,3,1), activation='relu', padding='same')(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    pool_drop_4 = layers.MaxPooling3D((1,1,2))(conv_5)
    if drop: pool_drop_4 = layers.Dropout(params["dropout"]["4"])(pool_drop_4)

    # second UP-convolutional layer: from (M/2,N/2,2) to (M,N,2)
    up_2 = layers.concatenate([layers.Conv3DTranspose(64, kernel_size=(2,2,1), strides=(2,2,1), activation='relu', padding='same')(pool_drop_4), conv_2], axis=3)
    conv_6 = layers.Conv3D(32, (3,3,1), activation='relu', padding='same')(up_2)
    conv_6 = layers.BatchNormalization()(conv_6)
    conv_6 = layers.Conv3D(32, (3,3,1), activation='relu', padding='same')(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    # from (M,N,2)  to (M,N,1)
    pool_drop_5 = layers.MaxPooling3D((1,1,2))(conv_6)
    if drop: pool_drop_5 = layers.Dropout(params["dropout"]["5"])(pool_drop_5)

    # last convolutional layer; plus reshape from (M,N,1) to (M,N)
    conv_7 = layers.Conv3D(1, (1,1,1), activation="sigmoid", padding='same')(pool_drop_5)
    y = layers.Reshape((constants.M,constants.N))(conv_7)

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
