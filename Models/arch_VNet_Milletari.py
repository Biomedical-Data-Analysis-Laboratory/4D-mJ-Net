import constants
from Utils import general_utils

from tensorflow.keras import layers, models, initializers
from tensorflow.keras.constraints import max_norm


################################################################################
# Model from Milletari V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params, to_categ):
    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9 / 5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

    input_x = layers.Input(shape=(constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1),
                           sparse=False)
    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    stage_1 = layers.Conv3D(16, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
    stage_1 = layers.PReLU()(stage_1)
    stage_1 = layers.Add()([input_x, stage_1])
    general_utils.print_int_shape(stage_1)  # (None, M, N, 30, 16)
    conv_1 = layers.Conv3D(32, kernel_size=(2, 2, 2), activation=None, padding='same', strides=(2, 2, 2),
                           kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_1)
    conv_1 = layers.PReLU()(conv_1)
    general_utils.print_int_shape(conv_1)  # (None, M/2, N/2, 15, 16)

    stage_2 = layers.Conv3D(32, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_1)
    stage_2 = layers.PReLU()(stage_2)
    stage_2 = layers.Conv3D(32, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_2)
    stage_2 = layers.PReLU()(stage_2)
    stage_2 = layers.Add()([conv_1, stage_2])
    general_utils.print_int_shape(stage_2)  # (None, M/2, N/2, 15, 32)
    conv_2 = layers.Conv3D(64, kernel_size=(2, 2, 2), activation=None, padding='same', strides=(2, 2, 3),
                           kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_2)
    conv_2 = layers.PReLU()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, M/4, N/4, 5, 32)

    stage_3 = layers.Conv3D(64, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_2)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Conv3D(64, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Conv3D(64, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init)(stage_3)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Add()([conv_2, stage_3])
    general_utils.print_int_shape(stage_3)  # (None, M/4, N/4, 5, 64)
    conv_3 = layers.Conv3D(128, kernel_size=(5, 5, 5), activation=None, padding='same', strides=(2, 2, 5),
                           kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_3)
    conv_3 = layers.PReLU()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, M/8, N/8, 1, 64)

    stage_4 = layers.Conv3D(128, kernel_size=(4, 4, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_3)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Conv3D(128, kernel_size=(4, 4, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Conv3D(128, kernel_size=(4, 4, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Add()([conv_3, stage_4])
    general_utils.print_int_shape(stage_4)  # (None, M/8, N/8, 1, 128)
    conv_4 = layers.Conv3D(256, kernel_size=(4, 4, 1), activation=None, padding='same', strides=(2, 2, 1),
                           kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_4)
    conv_4 = layers.PReLU()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, M/16, N/16, 1, 128)

    stage_5 = layers.Conv3D(256, kernel_size=(2, 2, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Conv3D(256, kernel_size=(2, 2, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Conv3D(256, kernel_size=(2, 2, 1), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Add()([conv_4, stage_5])
    general_utils.print_int_shape(stage_5)  # (None, M/16, N/16, 1, 256)
    conv_5 = layers.Conv3DTranspose(256, kernel_size=(2, 2, 2), activation=None, padding='same', strides=(2, 2, 1),
                                    kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_5)
    conv_5 = layers.PReLU()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, M/8, N/8, 1, 256)

    fine_grain_4 = layers.Conv3D(256, kernel_size=(1, 1, 1), activation=None, padding='same',
                                 kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_4)
    stage_6 = layers.Add()([fine_grain_4, conv_5])
    stage_6 = layers.Conv3D(256, kernel_size=(2, 2, 2), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_6)
    stage_6 = layers.PReLU()(stage_6)
    stage_6 = layers.Conv3D(256, kernel_size=(2, 2, 2), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_6)
    stage_6 = layers.PReLU()(stage_6)
    stage_6 = layers.Conv3D(256, kernel_size=(2, 2, 2), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_6)
    stage_6 = layers.PReLU()(stage_6)
    stage_6 = layers.Add()([stage_6, conv_5])
    general_utils.print_int_shape(stage_6)  # (None, M/8, N/8, 1, 128)
    conv_6 = layers.Conv3DTranspose(128, kernel_size=(2, 2, 2), activation=None, padding='same', strides=(2, 2, 5),
                                    kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_6)
    conv_6 = layers.PReLU()(conv_6)
    general_utils.print_int_shape(conv_6)  # (None, M/4, N/4, 5, 64)

    fine_grain_3 = layers.Conv3D(128, kernel_size=(1, 1, 1), activation=None, padding='same',
                                 kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_3)
    stage_7 = layers.Add()([fine_grain_3, conv_6])
    stage_7 = layers.Conv3D(128, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_7)
    stage_7 = layers.PReLU()(stage_7)
    stage_7 = layers.Conv3D(128, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_7)
    stage_7 = layers.PReLU()(stage_7)
    stage_7 = layers.Conv3D(128, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_7)
    stage_7 = layers.PReLU()(stage_7)
    stage_7 = layers.Add()([stage_7, conv_6])
    general_utils.print_int_shape(stage_7)  # (None, M/4, N/4, 5, 128)
    conv_7 = layers.Conv3DTranspose(64, kernel_size=(2, 2, 2), activation=None, padding='same', strides=(2, 2, 3),
                                    kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_7)
    conv_7 = layers.PReLU()(conv_7)
    general_utils.print_int_shape(conv_7)  # (None, M/2, N/2, 15, 64)

    fine_grain_2 = layers.Conv3D(64, kernel_size=(1, 1, 1), activation=None, padding='same',
                                 kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_2)
    stage_8 = layers.Add()([fine_grain_2, conv_7])
    stage_8 = layers.Conv3D(64, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_8)
    stage_8 = layers.PReLU()(stage_8)
    stage_8 = layers.Conv3D(64, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_8)
    stage_8 = layers.PReLU()(stage_8)
    stage_8 = layers.Add()([conv_7, stage_8])
    general_utils.print_int_shape(stage_8)  # (None, M/2, N/2, 15, 64)
    conv_8 = layers.Conv3DTranspose(32, kernel_size=(5, 5, 5), activation=None, padding='same', strides=(2, 2, 2),
                                    kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_8)
    conv_8 = layers.PReLU()(conv_8)
    general_utils.print_int_shape(conv_8)  # (None, M, N, 30, 32)

    fine_grain_1 = layers.Conv3D(32, kernel_size=(1, 1, 1), activation=None, padding='same',
                                 kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_1)
    stage_9 = layers.Add()([fine_grain_1, conv_8])
    stage_9 = layers.Conv3D(32, kernel_size=(5, 5, 5), activation=None, padding='same',
                            kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_9)
    stage_9 = layers.PReLU()(stage_9)
    stage_9 = layers.Add()([conv_8, stage_9])
    general_utils.print_int_shape(stage_9)  # (None, M, N, 30, 32)

    last_channels = 1
    activation_func = "sigmoid"
    shape_output = (constants.getM(), constants.getN())
    if to_categ:
        last_channels = len(constants.LABELS)
        activation_func = "softmax"
        shape_output = (constants.getM(), constants.getN(), last_channels)

    conv_9 = layers.Conv3D(last_channels, kernel_size=(5, 5, constants.NUMBER_OF_IMAGE_PER_SECTION),
                           activation=activation_func, padding='same', kernel_initializer=kernel_init,
                           strides=(1, 1, constants.NUMBER_OF_IMAGE_PER_SECTION),
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(stage_9)

    output = layers.Reshape(shape_output)(conv_9)
    general_utils.print_int_shape(output)  # (None, M, N, 4)

    model = models.Model(inputs=input_x, outputs=output)
    return model
