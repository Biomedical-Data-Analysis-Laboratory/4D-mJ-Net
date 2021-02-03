import constants
from Utils import general_utils


from keras import layers, regularizers, initializers, models
# from keras.applications import VGG16

def simple_autoencoder(params, to_categ):
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)

    activ_func = layers.LeakyReLU(alpha=0.33)
    l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)

    # # Create base model
    # base_model = VGG16(
    #     weights='imagenet',
    #     include_top=False
    # )
    # # Freeze base model
    # base_model.trainable = False

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)
    # x = base_model(input_x, training=False)
    # general_utils.print_int_shape(x)
    conv_1 = layers.Conv3D(16, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(input_x)
    conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)
    conv_2 = layers.Conv3D(8, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)
    pool_1 = layers.MaxPooling3D((2,2,2))(conv_2)
    general_utils.print_int_shape(pool_1)

    conv_3 = layers.Conv3D(8, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_1)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)
    conv_4 = layers.Conv3D(4, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_3)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)
    pool_2 = layers.MaxPooling3D((2,2,2))(conv_4)
    general_utils.print_int_shape(pool_2)

    conv_5 = layers.Conv3D(4, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_2)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)
    conv_6 = layers.Conv3D(8, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_5)
    conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6)
    up_1 = layers.UpSampling3D((2,2,2))(conv_6)
    general_utils.print_int_shape(up_1)

    conv_7 = layers.Conv3D(8, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_1)
    conv_7 = layers.BatchNormalization()(conv_7)
    general_utils.print_int_shape(conv_7)
    conv_8 = layers.Conv3D(16, kernel_size=(3,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_7)
    conv_8 = layers.BatchNormalization()(conv_8)
    general_utils.print_int_shape(conv_8)
    up_2 = layers.UpSampling3D((2,2,2))(conv_8)
    general_utils.print_int_shape(up_2)

    decoded = layers.Conv3D(1, (3,3,3), activation=activ_func, padding='same')(up_2)
    general_utils.print_int_shape(decoded)
    model = models.Model(inputs=input_x, outputs=decoded)

    return model
