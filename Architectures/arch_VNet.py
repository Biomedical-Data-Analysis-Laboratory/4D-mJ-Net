import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Model.constants import *
from Utils import general_utils

from tensorflow.keras import layers, models, initializers
import tensorflow_addons as tfa
#from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.constraints import max_norm


################################################################################
# Model from Milletari's V-Net (https://arxiv.org/pdf/1606.04797.pdf)
def VNet_Milletari(params):
    kernel_size_1, kernel_size_2 = (5,5,5), (2,2,2)
    channels = [8,16,32,64,128,256]
    channels = [int(ch/4) for ch in channels]

    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["bias_constraint"])

    input_shape = (get_m(), get_n(), getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(),
                                                                                                 get_m(), get_n(), 1)
    if get_USE_PM(): # use the PMs as input and concatenate them
        list_input = []
        if "multiInput" in params.keys():
            for pm in ["cbf", "cbv", "ttp", "mip", "mtt", "tmax"]:
                if pm in params["multiInput"].keys() and params["multiInput"][pm] == 1: list_input.append(layers.Input(shape=(get_m(),
                                                                                                                              get_n(), 3, 1), sparse=False))
        input_x = layers.Concatenate(3)(list_input)

    else: input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    # Stage 1
    stage_1 = layers.Conv3D(channels[1],kernel_size=kernel_size_1,activation=layers.PReLU(), padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(input_x)
    stage_1 = layers.Add()([input_x, stage_1])
    general_utils.print_int_shape(stage_1)  # (None, M, N, 30, 16)
    stride_1 = (2, 2, params["strides"]["conv.1"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.1"], 2, 2)
    conv_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=stride_1)(stage_1)
    general_utils.print_int_shape(conv_1)  # (None, M, N, 30, 16)

    # Stage 2
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_1)
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_2)
    stage_2 = layers.Add()([conv_1, stage_2])
    general_utils.print_int_shape(stage_2)  # (None, M/2, N/2, 15, 32)
    stride_2 = (2, 2, params["strides"]["conv.2"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.2"], 2, 2)
    conv_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=stride_2)(stage_2)
    general_utils.print_int_shape(conv_2)  # (None, M/2, N/2, 15, 32)

    # Stage 3
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_2)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.Add()([conv_2, stage_3])
    general_utils.print_int_shape(stage_3)  # (None, M/4, N/4, 5, 64)
    stride_3 = (2, 2, params["strides"]["conv.3"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.3"], 2, 2)
    conv_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_3,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)

    # Stage 4
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_3)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.Add()([conv_3, stage_4])
    general_utils.print_int_shape(stage_4)  # (None, M/8, N/8, 1, 128)
    stride_4 = (2, 2, params["strides"]["conv.4"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.4"], 2, 2)
    conv_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_4,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    general_utils.print_int_shape(conv_4)  # (None, M/16, N/16, 1, 128)

    # Stage 5
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_4)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.Add()([conv_4, stage_5])
    general_utils.print_int_shape(stage_5)  # (None, M/16, N/16, 1, 256)
    conv_5 = layers.Conv3DTranspose(channels[5],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_4,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)

    # R-Stage 4
    # fine-grained feature forwarding (== concatenation according to the U-Net paper)
    r_stage_4 = layers.Concatenate(-1)([stage_4, conv_5])
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Add()([r_stage_4, conv_5])
    general_utils.print_int_shape(r_stage_4)  # (None, M/8, N/8, 1, 128)
    conv_6 = layers.Conv3DTranspose(channels[4],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_3,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)

    # R-Stage 3
    r_stage_3 = layers.Concatenate(-1)([stage_3, conv_6])
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Add()([r_stage_3, conv_6])
    general_utils.print_int_shape(r_stage_3)  # (None, M/4, N/4, 5, 128)
    conv_7 = layers.Conv3DTranspose(channels[3],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_2,kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(r_stage_3)

    # R-Stage 2
    r_stage_2 = layers.Concatenate(-1)([stage_2, conv_7])
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.Add()([conv_7, r_stage_2])
    general_utils.print_int_shape(r_stage_2)  # (None, M/2, N/2, 15, 64)
    conv_8 = layers.Conv3DTranspose(channels[2],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_1,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)

    # R-Stage 1
    r_stage_1 = layers.Concatenate(-1)([stage_1, conv_8])
    r_stage_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    r_stage_1 = layers.Add()([conv_8, r_stage_1])
    general_utils.print_int_shape(r_stage_1)  # (None, M, N, 30, 32)

    last_channels = 1
    activation_func = "sigmoid"
    shape_output = (get_m(), get_n())
    if is_TO_CATEG():
        last_channels = len(get_labels())
        activation_func = "softmax"
        shape_output = (get_m(), get_n(), last_channels)

    stride_5 = (1, 1, params["strides"]["conv.1"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.1"], 1, 1)
    last_conv = layers.Conv3D(channels[2],kernel_size=stride_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_5,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    stride_6 = (1, 1, params["strides"]["conv.2"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.2"], 1, 1)
    last_conv = layers.Conv3D(channels[1],kernel_size=stride_2,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_6,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    stride_7 = (1, 1, params["strides"]["conv.3"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.3"], 1, 1)
    last_conv = layers.Conv3D(channels[0],kernel_size=stride_3,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_7,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    stride_8 = (1, 1, params["strides"]["conv.4"]) if is_timelast() or get_USE_PM() else (params["strides"]["conv.4"], 1, 1)
    last_conv = layers.Conv3D(channels[0],kernel_size=stride_4,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_8,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    last_conv = layers.Conv3D(last_channels,kernel_size=(1,1,1),activation=activation_func,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)

    output = layers.Reshape(shape_output)(last_conv)
    general_utils.print_int_shape(output)  # (None, M, N, 4)

    model = models.Model(inputs=input_x, outputs=output) if not get_USE_PM() else models.Model(inputs=list_input, outputs=output)
    return model


################################################################################
# Model of V-Net Light
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098643
def VNet_Light(params):
    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["bias_constraint"])

    input_shape = (get_m(), get_n(), getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(),
                                                                                                 get_m(), get_n(), 1)
    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    l0 = ThreeD_Light_Module(input_x, params["g"], 16, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l0_pool = layers.MaxPooling3D((2,2,2))(l0)
    general_utils.print_int_shape(l0_pool)

    l1 = ThreeD_Light_Module(l0_pool, 16, 32, params["g"], kernel_init, kernel_constraint, bias_constraint)
    pool_size = (2,2,3) if is_timelast() else (3, 2, 2)
    l1_pool = layers.MaxPooling3D(pool_size)(l1)
    general_utils.print_int_shape(l1_pool)

    l2 = ThreeD_Light_Module(l1_pool, 32, 64, params["g"], kernel_init, kernel_constraint, bias_constraint)
    pool_size = (2,2,5) if is_timelast() else (5, 2, 2)
    l2_pool = layers.MaxPooling3D(pool_size)(l2)
    general_utils.print_int_shape(l2_pool)

    l3 = ThreeD_Light_Module(l2_pool, 64, 128, params["g"], kernel_init, kernel_constraint, bias_constraint)
    pool_size = (2,2,1) if is_timelast() else (1, 2, 2)
    l3_pool = layers.MaxPooling3D(pool_size)(l3)
    general_utils.print_int_shape(l3_pool)

    l4 = ThreeD_Light_Module(l3_pool, 128, 256, params["g"], kernel_init, kernel_constraint, bias_constraint)
    pool_size = (2,2,1) if is_timelast() else (1, 2, 2)
    l4_pool = layers.MaxPooling3D(pool_size)(l4)
    general_utils.print_int_shape(l4_pool)

    l5 = ThreeD_Light_Module(l4_pool, 256, 512, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l5_size = (2,2,1) if is_timelast() else (1, 2, 2)
    l5_deconv = layers.Conv3DTranspose(256, kernel_size=l5_size, strides=l5_size)(l5)
    l5_conc = layers.concatenate([l5_deconv, l4], axis=-1)
    general_utils.print_int_shape(l5_conc)

    l6 = ThreeD_Light_Module(l5_conc, 128, 256, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l6_size = (2,2,1) if is_timelast() else (1, 2, 2)
    l6_deconv = layers.Conv3DTranspose(128, kernel_size=l6_size, strides=l6_size)(l6)
    l6_conc = layers.concatenate([l6_deconv, l3], axis=-1)
    general_utils.print_int_shape(l6_conc)

    l7 = ThreeD_Light_Module(l6_conc, 64, 128, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l7_size = (2,2,5) if is_timelast() else (5, 2, 2)
    l7_deconv = layers.Conv3DTranspose(64, kernel_size=l7_size, strides=l7_size)(l7)
    l7_conc = layers.concatenate([l7_deconv, l2], axis=-1)
    general_utils.print_int_shape(l7_conc)

    l8 = ThreeD_Light_Module(l7_conc, 32, 64, params["g"], kernel_init, kernel_constraint, bias_constraint)
    l8_size = (2,2,3) if is_timelast() else (3, 2, 2)
    l8_deconv = layers.Conv3DTranspose(32, kernel_size=l8_size, strides=l8_size)(l8)
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
    shape_output = (get_m(), get_n())
    if is_TO_CATEG():
        last_channels = len(get_labels())
        activation_func = "softmax"
        shape_output = (get_m(), get_n(), last_channels)

    k_size = (3,3,2) if is_timelast() else (2, 3, 3)
    stride = (1,1,2) if is_timelast() else (2, 1, 1)
    last_conv = layers.Conv3D(32,kernel_size=k_size,activation="relu",padding='same',kernel_initializer=kernel_init,
                              strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    last_conv = tfa.layers.InstanceNormalization()(last_conv)
    stride = (1,1,3) if is_timelast() else (3, 1, 1)
    last_conv = layers.Conv3D(16,kernel_size=(3,3,3),activation="relu",padding='same',kernel_initializer=kernel_init,
                              strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    last_conv = tfa.layers.InstanceNormalization()(last_conv)

    k_size = (3,3,5) if is_timelast() else (5, 3, 3)
    stride = (1,1,5) if is_timelast() else (5, 1, 1)
    last_conv = layers.Conv3D(8,kernel_size=k_size,activation="relu",padding='same',kernel_initializer=kernel_init,
                              strides=stride, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
    last_conv = tfa.layers.InstanceNormalization()(last_conv)
    last_conv = layers.Conv3D(last_channels,kernel_size=(1,1,1),activation=activation_func,padding='same',kernel_initializer=kernel_init,
                              strides=(1,1,1), kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(last_conv)
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


def VNet_Milletari_PMS(params, multiInput):
    kernel_size_1, kernel_size_2 = (5,5,5), (2,2,2)
    channels = [8,16,32,64,128,256]
    channels = [int(ch/4) for ch in channels]

    # Hu initializer
    kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.get_kernel_bias_constraint(
        params["bias_constraint"])

    list_input = []
    for pm in ["cbf", "cbv", "ttp", "mip", "mtt", "tmax"]:
        if pm in multiInput.keys() and multiInput[pm] == 1: list_input.append(layers.Input(shape=(get_m(), get_n(), 3), sparse=False))

    reshape_inputs = []
    for inp in list_input: reshape_inputs.append(layers.Reshape((get_m(), get_n(), 3, 1))(inp))
    input_x = layers.Concatenate(3)(reshape_inputs)

    general_utils.print_int_shape(input_x)  # (None, M, N, 30, 16)

    # Stage 1
    stage_1 = layers.Conv3D(channels[1],kernel_size=kernel_size_1,activation=layers.PReLU(), padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(input_x)
    stage_1 = layers.Add()([input_x, stage_1])
    general_utils.print_int_shape(stage_1)  # (None, M, N, 30, 16)
    stride_1 = (2, 2, params["strides"]["conv.1"])
    conv_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=stride_1)(stage_1)
    general_utils.print_int_shape(conv_1)  # (None, M, N, 30, 16)

    # Stage 2
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_1)
    stage_2 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_2)
    stage_2 = layers.Add()([conv_1, stage_2])
    general_utils.print_int_shape(stage_2)  # (None, M/2, N/2, 15, 32)
    stride_2 = (2, 2, params["strides"]["conv.2"])
    conv_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',bias_constraint=bias_constraint,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,strides=stride_2)(stage_2)
    general_utils.print_int_shape(conv_2)  # (None, M/2, N/2, 15, 32)

    # Stage 3
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_2)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)
    stage_3 = layers.Add()([conv_2, stage_3])
    general_utils.print_int_shape(stage_3)  # (None, M/4, N/4, 5, 64)
    stride_3 = (2, 2, params["strides"]["conv.3"])
    conv_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_3,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_3)

    # Stage 4
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_3)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    stage_4 = layers.Add()([conv_3, stage_4])
    general_utils.print_int_shape(stage_4)  # (None, M/8, N/8, 1, 128)
    stride_4 = (2, 2, params["strides"]["conv.4"])
    conv_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_4,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_4)
    general_utils.print_int_shape(conv_4)  # (None, M/16, N/16, 1, 128)

    # Stage 5
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(conv_4)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)
    stage_5 = layers.Add()([conv_4, stage_5])
    general_utils.print_int_shape(stage_5)  # (None, M/16, N/16, 1, 256)
    conv_5 = layers.Conv3DTranspose(channels[5],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_4,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(stage_5)

    # R-Stage 4
    # fine-grained feature forwarding (== concatenation according to the U-Net paper)
    r_stage_4 = layers.Concatenate(-1)([stage_4, conv_5])
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Conv3D(channels[5],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)
    r_stage_4 = layers.Add()([r_stage_4, conv_5])
    general_utils.print_int_shape(r_stage_4)  # (None, M/8, N/8, 1, 128)
    conv_6 = layers.Conv3DTranspose(channels[4],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_3,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_4)

    # R-Stage 3
    r_stage_3 = layers.Concatenate(-1)([stage_3, conv_6])
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Conv3D(channels[4],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_3)
    r_stage_3 = layers.Add()([r_stage_3, conv_6])
    general_utils.print_int_shape(r_stage_3)  # (None, M/4, N/4, 5, 128)
    conv_7 = layers.Conv3DTranspose(channels[3],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_2,kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(r_stage_3)

    # R-Stage 2
    r_stage_2 = layers.Concatenate(-1)([stage_2, conv_7])
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.Conv3D(channels[3],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)
    r_stage_2 = layers.Add()([conv_7, r_stage_2])
    general_utils.print_int_shape(r_stage_2)  # (None, M/2, N/2, 15, 64)
    conv_8 = layers.Conv3DTranspose(channels[2],kernel_size=kernel_size_2,activation=layers.PReLU(),padding='same',strides=stride_1,kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_2)

    # R-Stage 1
    r_stage_1 = layers.Concatenate(-1)([stage_1, conv_8])
    r_stage_1 = layers.Conv3D(channels[2],kernel_size=kernel_size_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    r_stage_1 = layers.Add()([conv_8, r_stage_1])
    general_utils.print_int_shape(r_stage_1)  # (None, M, N, 30, 32)

    last_channels = 1
    activation_func = "sigmoid"
    shape_output = (get_m(), get_n())
    if is_TO_CATEG():
        last_channels = len(get_labels())
        activation_func = "softmax"
        shape_output = (get_m(), get_n(), last_channels)

    stride_5 = (1, 1, params["strides"]["conv.1"])
    last_conv = layers.Conv3D(channels[2],kernel_size=stride_1,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_5,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(r_stage_1)
    stride_6 = (1, 1, params["strides"]["conv.2"])
    last_conv = layers.Conv3D(channels[1],kernel_size=stride_2,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_6,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    stride_7 = (1, 1, params["strides"]["conv.3"])
    last_conv = layers.Conv3D(channels[0],kernel_size=stride_3,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_7,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    stride_8 = (1, 1, params["strides"]["conv.4"])
    last_conv = layers.Conv3D(channels[0],kernel_size=stride_4,activation=layers.PReLU(),padding='same',kernel_initializer=kernel_init,strides=stride_8,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)
    last_conv = layers.Conv3D(last_channels,kernel_size=(1,1,1),activation=activation_func,padding='same',kernel_initializer=kernel_init,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(last_conv)

    output = layers.Reshape(shape_output)(last_conv)
    general_utils.print_int_shape(output)  # (None, M, N, 4)

    model = models.Model(inputs=list_input, outputs=output)
    return model