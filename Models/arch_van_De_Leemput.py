import constants
from Utils import general_utils, spatial_pyramid

from tensorflow.keras import layers, models, regularizers, initializers
import tensorflow.keras.backend as K

################################################################################
# Model from Van De Leemput (https://doi.org/10.1109/ACCESS.2019.2910348)
def van_De_Leemput(X, params, to_categ):
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

    conv_15 = layers.Conv3D(len(constants.LABELS), (1,1,1), activation="softmax", padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=hu_init)(add_7)
    print(K.int_shape(conv_15)) # (None, 1, 16, 16, 4)

    y = layers.Reshape((constants.getM(),constants.getN(),4))(conv_15)
    print(K.int_shape(y))
    model = models.Model(inputs=input_x, outputs=y)
    return model
