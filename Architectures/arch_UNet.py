from Model import constants
from Utils import general_utils

from keras import layers, models, regularizers, initializers
import tensorflow.keras.backend as K

################################################################################
# Model from Ronneberger (original paper of U-Net) (https://doi.org/10.1007/978-3-319-24574-4_28)
def Ronneberger_UNET(params):
    # Hu initializer = [0, sqrt(2/fan_in)]
    hu_init = initializers.he_normal(seed=None)

    input_x = layers.Input(shape=(constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1), sparse=False)
    print(K.int_shape(input_x)) # (None, 30, 16, 16, 1)
    conv_1 = layers.Conv3D(64, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION, 3, 3), activation='relu', padding='same', kernel_initializer=hu_init)(input_x)
    print(K.int_shape(conv_1)) # (None, 30, 16, 16, 64)
    conv_2 = layers.Conv3D(64, kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION, 3, 3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_1)
    print(K.int_shape(conv_2)) # (None, 30, 16, 16, 64)

    pool_1 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION, 2, 2))(conv_2)
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

    # up_1 = layers.Conv3DTranspose(512, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_10)
    up_1 = layers.UpSampling3D(size=(1,2,2))(conv_10)
    print(K.int_shape(up_1)) # (None, 1, 2, 2, 1024)
    addconv_8 = layers.concatenate([conv_8, conv_8])
    print(K.int_shape(addconv_8)) # (None, 1, 2, 2, 1024)
    conc_1 = layers.concatenate([up_1, addconv_8], axis=-1)
    # conc_1 = layers.concatenate([up_1, conv_8], axis=-1)
    print(K.int_shape(conc_1)) # (None, 1, 2, 2, 1024)
    conv_11 = layers.Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_1)
    print(K.int_shape(conv_11)) # (None, 1, 2, 2, 512)
    conv_12 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_11)
    print(K.int_shape(conv_12)) # (None, 1, 2, 2, 256)

    # up_2 = layers.Conv3DTranspose(256, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_12)
    up_2 = layers.UpSampling3D(size=(1,2,2))(conv_12)
    print(K.int_shape(up_1)) # (None, 1, 4, 4, 256)
    conc_2 = layers.concatenate([up_2, conv_6], axis=-1)
    print(K.int_shape(conc_2)) # (None, 1, 4, 4, 256)
    conv_13 = layers.Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_2)
    print(K.int_shape(conv_13)) # (None, 1, 4, 4, 512)
    conv_14 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_13)
    print(K.int_shape(conv_14)) # (None, 1, 4, 4, 256)

    # up_3 = layers.Conv3DTranspose(128, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_14)
    up_3 = layers.UpSampling3D(size=(1,2,2))(conv_14)
    print(K.int_shape(up_3)) # (None, 1, 8, 8, 256)
    addconv_4 = layers.concatenate([conv_4, conv_4])
    print(K.int_shape(addconv_4)) # (None, 1, 8, 8, 256)
    conc_3 = layers.concatenate([up_3, addconv_4], axis=-1)
    # conc_3 = layers.concatenate([up_3, conv_4], axis=-1)
    print(K.int_shape(conc_3)) # (None, 1, 8, 8, 256)
    conv_15 = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_3)
    print(K.int_shape(conv_15)) # (None, 1, 8, 8, 256)
    conv_16 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_15)
    print(K.int_shape(conv_16)) # (None, 1, 8, 8, 128)

    # up_4 = layers.Conv3DTranspose(64, kernel_size=(1,2,2), strides=(1,2,2), activation='relu', padding='same', kernel_initializer=hu_init)(conv_16)
    up_4 = layers.UpSampling3D(size=(1,2,2))(conv_16)
    print(K.int_shape(up_4)) # (None, 1, 16, 16, 128)
    convpool_2 = layers.MaxPooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION, 1, 1))(conv_2)
    convpool_2 = layers.concatenate([convpool_2,convpool_2])
    conc_4 = layers.concatenate([up_4, convpool_2], axis=-1)
    print(K.int_shape(conc_4)) # (None, 1, 16, 16, 128)
    conv_17 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conc_4)
    print(K.int_shape(conv_17)) # (None, 1, 16, 16, 128)
    conv_18 = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_17)
    print(K.int_shape(conv_18)) # (None, 1, 16, 16, 64)
    conv_19 = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', kernel_initializer=hu_init)(conv_18)
    print(K.int_shape(conv_19)) # (None, 1, 16, 16, 64)

    conv_20 = layers.Conv3D(4, (1,1,1), activation="softmax", padding='same', kernel_initializer=hu_init)(conv_19)
    print(K.int_shape(conv_20)) # (None, 1, 16, 16, 4)

    y = layers.Reshape((constants.getM(), constants.getN(), 4))(conv_20)
    print(K.int_shape(y))
    model = models.Model(inputs=input_x, outputs=y)
    return model
