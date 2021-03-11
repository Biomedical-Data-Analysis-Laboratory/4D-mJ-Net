import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Model import constants
from Utils import general_utils

from tensorflow.keras import layers, models, initializers
import tensorflow_addons as tfa
#from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.constraints import max_norm


################################################################################
# Model from Milletari's V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params, to_categ):
    kernel_size_1, kernel_size_2 = (5,5,5), (2,2,2)
    channels = [8,16,32,64,128,256]
    channels = [int(ch / 2) for ch in channels]

    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint, bias_constraint = None, None  # max_norm(2.), max_norm(2.)

    input_x = layers.Input(shape=(constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1), sparse=False)
    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    # Stage 1
    stage_1 = layers.Conv3D(channels[1],kernel_size=kernel_size_1,activation=None, padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(input_x)
    stage_1 = layers.PReLU()(stage_1)
    stage_1 = layers.Add()([input_x, stage_1])
    general_utils.print_int_shape(stage_1)  # (None, M, N, 30, 16)
    conv_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_2,activation=None,padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=(2,2,params["strides"]["conv.1"]))(stage_1)
    conv_1 = layers.PReLU()(conv_1)
    general_utils.print_int_shape(conv_1)  # (None, M, N, 30, 16)

    # Stage 2
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_1)
    stage_2 = layers.PReLU()(stage_2)
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_2)
    stage_2 = layers.PReLU()(stage_2)
    stage_2 = layers.Add()([conv_1, stage_2])
    general_utils.print_int_shape(stage_2)  # (None, M/2, N/2, 15, 32)
    conv_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_2,activation=None,padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=(2,2,params["strides"]["conv.2"]))(stage_2)
    conv_2 = layers.PReLU()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, M/2, N/2, 15, 32)

    # Stage 3
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_2)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.PReLU()(stage_3)
    stage_3 = layers.Add()([conv_2, stage_3])
    general_utils.print_int_shape(stage_3)  # (None, M/4, N/4, 5, 64)
    conv_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.3"]),kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    conv_3 = layers.PReLU()(conv_3)

    # Stage 4
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_3)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.PReLU()(stage_4)
    stage_4 = layers.Add()([conv_3, stage_4])
    general_utils.print_int_shape(stage_4)  # (None, M/8, N/8, 1, 128)
    conv_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.4"]),kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    conv_4 = layers.PReLU()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, M/16, N/16, 1, 128)

    # Stage 5
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_4)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.PReLU()(stage_5)
    stage_5 = layers.Add()([conv_4, stage_5])
    general_utils.print_int_shape(stage_5)  # (None, M/16, N/16, 1, 256)
    conv_5 = layers.Conv3DTranspose(channels[5],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.4"]),kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    conv_5 = layers.PReLU()(conv_5)

    # R-Stage 4
    # fine-grained feature forwarding (== concatenation according to the U-Net paper
    r_stage_4 = layers.Concatenate(-1)([stage_4, conv_5])
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.PReLU()(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.PReLU()(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.PReLU()(r_stage_4)
    r_stage_4 = layers.Add()([r_stage_4, conv_5])
    general_utils.print_int_shape(r_stage_4)  # (None, M/8, N/8, 1, 128)
    conv_6 = layers.Conv3DTranspose(channels[4],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.3"]),kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    conv_6 = layers.PReLU()(conv_6)

    # R-Stage 3
    r_stage_3 = layers.Concatenate(-1)([stage_3, conv_6])
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.PReLU()(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.PReLU()(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.PReLU()(r_stage_3)
    r_stage_3 = layers.Add()([r_stage_3, conv_6])
    general_utils.print_int_shape(r_stage_3)  # (None, M/4, N/4, 5, 128)
    conv_7 = layers.Conv3DTranspose(channels[3],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.2"]),kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(r_stage_3)
    conv_7 = layers.PReLU()(conv_7)

    # R-Stage 2
    r_stage_2 = layers.Concatenate(-1)([stage_2, conv_7])
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.PReLU()(r_stage_2)
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.PReLU()(r_stage_2)
    r_stage_2 = layers.Add()([conv_7, r_stage_2])
    general_utils.print_int_shape(r_stage_2)  # (None, M/2, N/2, 15, 64)
    conv_8 = layers.Conv3DTranspose(channels[2],kernel_size=kernel_size_2,activation=None,padding='same',strides=(2,2,params["strides"]["conv.1"]),kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    conv_8 = layers.PReLU()(conv_8)

    # R-Stage 1
    r_stage_1 = layers.Concatenate(-1)([stage_1, conv_8])
    r_stage_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=None,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    r_stage_1 = layers.PReLU()(r_stage_1)
    r_stage_1 = layers.Add()([conv_8, r_stage_1])
    general_utils.print_int_shape(r_stage_1)  # (None, M, N, 30, 32)

    last_channels = 1
    activation_func = "sigmoid"
    shape_output = (constants.getM(), constants.getN())
    if to_categ:
        last_channels = len(constants.LABELS)
        activation_func = "softmax"
        shape_output = (constants.getM(), constants.getN(), last_channels)

    last_conv = layers.Conv3D(channels[2],kernel_size=(2,2,params["strides"]["conv.1"]),activation=None,padding='same',kernel_initializer=kernel_init,strides=(1,1,params["strides"]["conv.1"]),kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    last_conv = layers.PReLU()(last_conv)
    last_conv = layers.Conv3D(channels[1],kernel_size=(2,2,params["strides"]["conv.2"]),activation=None,padding='same',kernel_initializer=kernel_init,strides=(1,1,params["strides"]["conv.2"]),kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    last_conv = layers.PReLU()(last_conv)
    last_conv = layers.Conv3D(channels[0],kernel_size=(2,2,params["strides"]["conv.3"]),activation=None,padding='same',kernel_initializer=kernel_init,strides=(1,1,params["strides"]["conv.3"]),kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    last_conv = layers.PReLU()(last_conv)
    last_conv = layers.Conv3D(channels[0],kernel_size=(1,1,params["strides"]["conv.4"]),activation=None,padding='same',kernel_initializer=kernel_init,strides=(1,1,params["strides"]["conv.4"]),kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    last_conv = layers.PReLU()(last_conv)
    last_conv = layers.Conv3D(last_channels,kernel_size=(1,1,1),activation=activation_func,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)

    output = layers.Reshape(shape_output)(last_conv)
    general_utils.print_int_shape(output)  # (None, M, N, 4)

    model = models.Model(inputs=input_x, outputs=output)
    return model


################################################################################
# Model of V-Net Light
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098643
def VNet_Light(params, to_categ):
    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint, bias_constraint = None, None  # max_norm(2.), max_norm(2.)

    input_x = layers.Input(shape=(constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1), sparse=False)
    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    l0 = ThreeD_Light_Module(input_x, params["g"], 16, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l0_pool = layers.MaxPooling3D((2,2,2))(l0)
    general_utils.print_int_shape(l0_pool)

    l1 = ThreeD_Light_Module(l0_pool, 16, 32, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l1_pool = layers.MaxPooling3D((2,2,3))(l1)
    general_utils.print_int_shape(l1_pool)

    l2 = ThreeD_Light_Module(l1_pool, 32, 64, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l2_pool = layers.MaxPooling3D((2,2,5))(l2)
    general_utils.print_int_shape(l2_pool)

    l3 = ThreeD_Light_Module(l2_pool, 64, 128, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l3_pool = layers.MaxPooling3D((2,2,1))(l3)
    general_utils.print_int_shape(l3_pool)

    l4 = ThreeD_Light_Module(l3_pool, 128, 256, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l4_pool = layers.MaxPooling3D((2,2,1))(l4)
    general_utils.print_int_shape(l4_pool)

    l5 = ThreeD_Light_Module(l4_pool, 256, 512, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l5_deconv = layers.Conv3DTranspose(256, kernel_size=(2,2,1), strides=(2,2,1))(l5)
    l5_conc = layers.concatenate([l5_deconv, l4], axis=-1)
    general_utils.print_int_shape(l5_conc)

    l6 = ThreeD_Light_Module(l5_conc, 128, 256, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l6_deconv = layers.Conv3DTranspose(128, kernel_size=(2,2,1), strides=(2,2,1))(l6)
    l6_conc = layers.concatenate([l6_deconv, l3], axis=-1)
    general_utils.print_int_shape(l6_conc)

    l7 = ThreeD_Light_Module(l6_conc, 64, 128, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l7_deconv = layers.Conv3DTranspose(64, kernel_size=(2,2,5), strides=(2,2,5))(l7)
    l7_conc = layers.concatenate([l7_deconv, l2], axis=-1)
    general_utils.print_int_shape(l7_conc)

    l8 = ThreeD_Light_Module(l7_conc, 32, 64, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l8_deconv = layers.Conv3DTranspose(32, kernel_size=(2,2,3), strides=(2,2,3))(l8)
    l8_conc = layers.concatenate([l8_deconv, l1], axis=-1)
    general_utils.print_int_shape(l8_conc)

    l9 = ThreeD_Light_Module(l8_conc, 16, 32, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l9_deconv = layers.Conv3DTranspose(16, kernel_size=(2,2,2), strides=(2,2,2))(l9)
    l9_conc = layers.concatenate([l9_deconv, l0], axis=-1)
    general_utils.print_int_shape(l9_conc)

    last_conv = layers.Conv3D(params["g"], kernel_size=(3,3,3), activation="relu", padding='same', kernel_initializer=kernel_init,
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(l9_conc)
    last_conv = tfa.layers.InstanceNormalization()(last_conv)

    last_channels = 1
    activation_func = "sigmoid"
    shape_output = (constants.getM(), constants.getN())
    if to_categ:
        last_channels = len(constants.LABELS)
        activation_func = "softmax"
        shape_output = (constants.getM(), constants.getN(), last_channels)

    last_conv = layers.Conv3D(last_channels, kernel_size=(3, 3, constants.NUMBER_OF_IMAGE_PER_SECTION),
                              activation=activation_func, padding='same', kernel_initializer=kernel_init,
                              strides=(1, 1, constants.NUMBER_OF_IMAGE_PER_SECTION),
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    last_conv = tfa.layers.InstanceNormalization()(last_conv)

    output = layers.Reshape(shape_output)(last_conv)
    general_utils.print_int_shape(output)  # (None, M, N, 4)

    model = models.Model(inputs=input_x, outputs=output)
    return model


################################################################################
# Function that replicates the 3D light module for the V-Net light
def ThreeD_Light_Module(layer_in, Cin, Cout, g, kernel_init, kernel_constraint, bias_constraint):
    layer_out = []
    for ig in range(g):
        one = layers.Conv3D(int(Cin/g), kernel_size=(1,1,1), activation="relu", padding='same', kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(layer_in)
        one = tfa.layers.InstanceNormalization()(one)
        two = layers.Conv3D(int(Cin/g), kernel_size=(3,3,3), activation="relu", padding='same', kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(one)
        two = tfa.layers.InstanceNormalization()(two)
        layer_out.append(two)
        two_par = layers.Conv3D(int(Cin/g), kernel_size=(1,1,1), activation="relu", padding='same', kernel_initializer=kernel_init,
                                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(one)
        two_par = tfa.layers.InstanceNormalization()(two_par)
        layer_out.append(two_par)

    out = layers.concatenate(layer_out, axis=-1)
    assert out.shape[-1] == Cout

    return out
