from Model import constants
from Utils import general_utils, model_utils

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv3D, Conv2DTranspose, Conv3DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16


################################################################################
# mJ-Net model
def mJNet(params, batch=True, drop=False, longJ=False, v2=False):
    # from (30,M,N) to (1,M,N)

    size_two = (2,2,1) if constants.getTIMELAST() else (1,2,2)
    kernel_size = (3,3,1) if constants.getTIMELAST() else (1,3,3)
    activ_func = 'relu'
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    channels = [16,32,16,32,16,32,16,32,64,64,128,128,256,-1,-1,-1,-1,128,128,64,64,32,16]
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN(), 1)
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.getKernelInit(params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["bias_constraint"])

    if v2:  # version 2
        kernel_size = (3,3)
        activ_func = None
        # channels = [16,32,32,64,64,128,128,32,64,128,256,512,1024,512,1024,512,1024,-1,512,256,-1,128,64]
        channels = [16,16,32,32,64,64,-1,64,64,128,128,128,128,128,128,128,128,-1,128,128,-1,64,32]
        channels = [int(ch/4) for ch in channels]  # implemented due to memory issues

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)  # (None, 30, M, N, 1)

    if longJ:
        conv_01 = model_utils.convolutionLayer(input_x,channels[0],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 16)
        conv_01 = model_utils.convolutionLayer(conv_01,channels[1],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 32)

        pool_shape = (1,1,params["max_pool"]["long.1"]) if constants.getTIMELAST() else (params["max_pool"]["long.1"],1,1)
        pool_drop_01 = layers.MaxPooling3D(pool_shape)(conv_01)
        general_utils.print_int_shape(pool_drop_01)  # (None, 15, M, N, 32)
        conv_02 = model_utils.convolutionLayer(pool_drop_01,channels[2],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 32)
        conv_02 = model_utils.convolutionLayer(conv_02,channels[3],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 64)

        pool_shape = (1,1,params["max_pool"]["long.2"]) if constants.getTIMELAST() else (params["max_pool"]["long.2"],1,1)
        pool_drop_02 = layers.MaxPooling3D(pool_shape)(conv_02)
        general_utils.print_int_shape(pool_drop_02)  # (None, 5, M, N, 64)
        conv_03 = model_utils.convolutionLayer(pool_drop_02,channels[4],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 64)
        conv_03 = model_utils.convolutionLayer(conv_03,channels[5],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 128)

        pool_shape = (1,1,params["max_pool"]["long.3"]) if constants.getTIMELAST() else (params["max_pool"]["long.3"],1,1)
        pool_drop_1 = layers.MaxPooling3D(pool_shape)(conv_03)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["long.1"])(pool_drop_1)
    else:
        kernel_shape = (3,3,constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,3,3)
        conv_1 = model_utils.convolutionLayer(input_x,channels[6],kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
        if batch: conv_1 = layers.BatchNormalization()(conv_1)
        general_utils.print_int_shape(conv_1)  # (None, 30, M, N, 128)

        pool_shape = (1,1,constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,1,1)
        pool_drop_1 = layers.AveragePooling3D(pool_shape)(conv_1)
        # pool_drop_1 = spatial_pyramid.SPP3D([1,2,4], input_shape=(channels[6],None,None,None))(conv_1)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["1"])(pool_drop_1)

    conv_list = []
    input_conv_layer = pool_drop_1
    loop = 1
    while K.int_shape(input_conv_layer)[2]>32 and K.int_shape(input_conv_layer)[3]>32:
        conv_x = model_utils.convolutionLayer(input_conv_layer, channels[7] * loop, kernel_size, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=v2, timedistr=v2, is2D=v2)
        if batch: conv_x = layers.BatchNormalization()(conv_x)
        general_utils.print_int_shape(conv_x)  # (None, 1, M, N, 32)
        conv_x = model_utils.convolutionLayer(conv_x,channels[8]*loop,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_x = layers.BatchNormalization()(conv_x)
        general_utils.print_int_shape(conv_x)  # (None, 1, M, N, 64)
        input_conv_layer = layers.MaxPooling3D(size_two)(conv_x)
        general_utils.print_int_shape(input_conv_layer)  # (None, 1, M/2, N/2, 64)
        if drop: input_conv_layer = Dropout(params["dropout"]["loop"])(input_conv_layer)
        conv_list.append(conv_x)
        loop+=loop

    conv_4 = model_utils.convolutionLayer(input_conv_layer, channels[11], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=v2, timedistr=v2, is2D=v2)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 512)
    conv_4 = model_utils.convolutionLayer(conv_4,channels[12],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 1024)

    if v2:
        kernel_size = (3,3,1) if constants.getTIMELAST() else (1,3,3)
        pool_drop_3_1 = layers.MaxPooling3D(size_two)(conv_4)
        general_utils.print_int_shape(pool_drop_3_1)  # (None, 1, M/8, N/8, 1024)
        conv_4_1 = model_utils.convolutionLayer(pool_drop_3_1,channels[13],3,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_4_1 = layers.BatchNormalization()(conv_4_1)
        general_utils.print_int_shape(conv_4_1)  # (None, 1, M/8, N/8, 512)
        conv_5_1 = model_utils.convolutionLayer(conv_4_1,channels[14],3,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2,timedistr=v2,is2D=v2)
        if batch: conv_5_1 = layers.BatchNormalization()(conv_5_1)
        if drop: conv_5_1 = Dropout(params["dropout"]["3.1"])(conv_5_1)
        general_utils.print_int_shape(conv_5_1)  # (None, 1, M/8, N/8, 1024)

        attGate_1 = model_utils.attentionGateBlock(x=conv_4, g=conv_5_1, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_0 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_5_1), attGate_1], axis=-1)

        conv_6_1 = model_utils.convolutionLayer(up_0,channels[15],3,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
        if batch: conv_6_1 = layers.BatchNormalization()(conv_6_1)
        general_utils.print_int_shape(conv_6_1)  # (None, 1, M/4, N/4, 512)
        conv_7_1 = model_utils.convolutionLayer(conv_6_1,channels[16],3,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
        if batch: conv_7_1 = layers.BatchNormalization()(conv_7_1)
        general_utils.print_int_shape(conv_7_1)  # (None, 1, M/4, N/4, 1024)

        attGate_2 = model_utils.attentionGateBlock(x=conv_3, g=conv_7_1, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_1 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_7_1), attGate_2], axis=-1)
    else:
        # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
        axis = 3 if constants.getTIMELAST() else -1
        up_1 = layers.concatenate([
            Conv3DTranspose(channels[17], kernel_size=size_two, strides=size_two, activation=activ_func,
                            padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4),
            conv_list.pop()], axis=axis)

    input_conv_layer = up_1
    loop = 1
    while K.int_shape(input_conv_layer)[2]<constants.getM() and K.int_shape(input_conv_layer)[3]<constants.getN():
        general_utils.print_int_shape(input_conv_layer)  # (None, 1, M/2, N/2, 1024)
        conv_x = model_utils.convolutionLayer(input_conv_layer, channels[18] * loop, kernel_size, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=v2)
        if batch: conv_x = layers.BatchNormalization()(conv_x)
        general_utils.print_int_shape(conv_x)  # (None, 1, M/2, N/2, 512)
        conv_x = model_utils.convolutionLayer(conv_x,channels[19]*loop,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
        if batch: conv_x = layers.BatchNormalization()(conv_x)
        general_utils.print_int_shape(conv_x)  # (None, 1, M/2, N/2, 256)

        # TODO: check v2!
        if v2:
            attGate_3 = model_utils.attentionGateBlock(x=conv_list.pop(), g=conv_x, inter_shape=128, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                     kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
            up_2 = layers.concatenate([layers.UpSampling3D(size=size_two)(conv_x), attGate_3], axis=-1)
        else:
            # pool_shape = (1,1,2) if constants.getTIMELAST() else (2,1,1)
            # pool_drop_4 = layers.MaxPooling3D(pool_shape)(conv_x)
            # general_utils.print_int_shape(pool_drop_4)  # (None, 1, M/2, N/2, 512)
            if drop: conv_x = Dropout(params["dropout"]["loop"])(conv_x)
            # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
            axis = 3 if constants.getTIMELAST() else -1
            up_2 = layers.concatenate([
                Conv3DTranspose(channels[20], kernel_size=size_two, strides=size_two, activation=activ_func,
                                padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_x),
                conv_list.pop()], axis=axis)
            input_conv_layer = up_2
            loop += loop

    general_utils.print_int_shape(input_conv_layer)  # (None, X, M, N, 1024)
    conv_6 = model_utils.convolutionLayer(input_conv_layer,channels[21],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
    if batch: conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6)  # (None, X, M, N, 128)
    conv_6 = model_utils.convolutionLayer(conv_6,channels[22],kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=v2)
    pool_drop_5 = layers.BatchNormalization()(conv_6) if batch else conv_6
    general_utils.print_int_shape(pool_drop_5)  # (None, X, M, N, 64)

    if not v2:
        # from (2,M,N)  to (1,M,N)
        # pool_shape = (1,1,2) if constants.getTIMELAST() else (2,1,1)
        # pool_drop_5 = layers.MaxPooling3D(pool_shape)(pool_drop_5)
        # general_utils.print_int_shape(pool_drop_5)  # (None, 1, M, N, 16)
        if drop: pool_drop_5 = Dropout(params["dropout"]["5"])(pool_drop_5)

    # set the softmax activation function if the flag is set
    act_name = "softmax" if constants.getTO_CATEG() else "sigmoid"
    n_chann = len(constants.LABELS) if constants.getTO_CATEG() else 1
    shape_output = (constants.getM(), constants.getN(), n_chann) if constants.getTO_CATEG() else (constants.getM(), constants.getN())

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = model_utils.convolutionLayer(pool_drop_5,n_chann,(1,1,1),act_name,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=False)
    general_utils.print_int_shape(conv_7)  # (None, 1, M, N, 1)
    y = layers.Reshape(shape_output)(conv_7)
    general_utils.print_int_shape(y)  # (None, M, N)
    model = models.Model(inputs=input_x, outputs=y)

    return model


################################################################################
# mJ-Net model version 3D ?
def mJNet_2D_with_VGG16(params, multiInput, batch=True, drop=True, leaky=True, attentiongate=True):
    kernel_size, size_two = (3,3), (2,2)
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN(), 1)
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    activ_func = None if leaky else 'relu'
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.getKernelInit(params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["bias_constraint"])

    x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(x)

    kernel_shape = (3,3,constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,3,3)
    conv_1 = model_utils.convolutionLayer(x,32,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)

    kernel_shape = (3,3,constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,3,3)
    strides_shape = (1,1,params["strides"]["conv.1"]) if constants.getTIMELAST() else (params["strides"]["conv.1"],1,1)
    conv_1 = model_utils.convolutionLayer(conv_1,32,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,strides=strides_shape,leaky=leaky)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)

    new_z = constants.NUMBER_OF_IMAGE_PER_SECTION/params["strides"]["conv.1"]
    kernel_shape = (3,3,int(new_z)) if constants.getTIMELAST() else (int(new_z),3,3)
    conv_2 = model_utils.convolutionLayer(conv_1,32,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)

    strides_shape = (1,1,params["strides"]["conv.2"]) if constants.getTIMELAST() else (params["strides"]["conv.2"],1,1)
    conv_2 = model_utils.convolutionLayer(conv_2,16,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,strides=strides_shape,leaky=leaky)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)

    new_z /= params["strides"]["conv.2"]
    kernel_shape = (3,3,int(new_z)) if constants.getTIMELAST() else (int(new_z),3,3)
    conv_3 = model_utils.convolutionLayer(conv_2,8,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)

    strides_shape = (1,1,params["strides"]["conv.3"]) if constants.getTIMELAST() else (params["strides"]["conv.3"],1,1)
    conv_3 = model_utils.convolutionLayer(conv_3,1,kernel_shape,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,strides=strides_shape,leaky=leaky)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)

    conv_3 = layers.Reshape(conv_3.shape[1:-1])(conv_3)
    general_utils.print_int_shape(conv_3)

    if conv_3.shape[-1]==1: conv_3 = Concatenate(-1)([conv_3,conv_3,conv_3])
    general_utils.print_int_shape(conv_3)
    # Add the VGG-16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=conv_3.shape[1:])
    # Freeze layers VGG-16 model
    vgg16_model.trainable = False if params["trainable"] == 0 else True
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg16_model.layers])

    conv_4 = model_utils.convolutionLayer(vgg16_model.output,128,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky,is2D=True)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    conv_4 = model_utils.convolutionLayer(conv_4,128,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky,is2D=True)
    if batch: conv_4 = layers.BatchNormalization()(conv_4)
    if drop: conv_4 = Dropout(params["dropout"]["conv.4"])(conv_4)

    inputs, conv_out = model_utils.addMoreInfo(multiInput, x, [conv_4])
    if len(conv_out)>1: conv_out = Concatenate(-1)(conv_out)
    elif len(conv_out)==1: conv_out = conv_out[0]

    if attentiongate:
        attGate_1 = model_utils.attentionGateBlock(x=layer_dict["block5_conv3"].output, g=conv_out, inter_shape=256, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, is2D=True)
        up_1 = layers.concatenate([layers.UpSampling2D(size=size_two)(conv_out), attGate_1], axis=-1)
        up_2 = model_utils.upSamplingPlusAttention(up_1,layer_dict["block4_conv3"].output,[128,128,128],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_3 = model_utils.upSamplingPlusAttention(up_2,layer_dict["block3_conv3"].output,[64,64,64],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_4 = model_utils.upSamplingPlusAttention(up_3,layer_dict["block2_conv2"].output,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
        up_5 = model_utils.upSamplingPlusAttention(up_4,layer_dict["block1_conv2"].output,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky, is2D=True)
    else:
        transp_1 = Conv2DTranspose(256, kernel_size=size_two, strides=size_two, activation=activ_func, padding='same',
                                   kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_out)
        if leaky: transp_1 = layers.LeakyReLU(alpha=0.33)(transp_1)
        if batch: transp_1 = layers.BatchNormalization()(transp_1)
        up_1 = Concatenate(-1)([transp_1, layer_dict["block5_conv3"].output])
        # going up with the layers
        up_2 = model_utils.upLayers(up_1, layer_dict["block4_conv3"].output, [128,128,128], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_3 = model_utils.upLayers(up_2, layer_dict["block3_conv3"].output, [64,64,64], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_4 = model_utils.upLayers(up_3, layer_dict["block2_conv2"].output, [32,32,32], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)
        up_5 = model_utils.upLayers(up_4, layer_dict["block1_conv2"].output, [16,16,16], kernel_size, size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky=True, is2D=True)

    general_utils.print_int_shape(up_1)
    general_utils.print_int_shape(up_2)
    general_utils.print_int_shape(up_3)
    general_utils.print_int_shape(up_4)
    general_utils.print_int_shape(up_5)

    final_conv_1 = model_utils.convolutionLayer(up_5,16,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky,is2D=True)
    if batch: final_conv_1 = layers.BatchNormalization()(final_conv_1)
    general_utils.print_int_shape(final_conv_1)
    final_conv_2 = model_utils.convolutionLayer(final_conv_1,16,kernel_size,activ_func,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,leaky=leaky,is2D=True)
    if batch: final_conv_2 = layers.BatchNormalization()(final_conv_2)
    general_utils.print_int_shape(final_conv_2)

     # set the softmax activation function if the flag is set
    act_name = "softmax" if constants.getTO_CATEG() else  "sigmoid"
    n_chann = len(constants.LABELS) if constants.getTO_CATEG() else 1
    shape_output = (constants.getM(), constants.getN(), n_chann) if constants.getTO_CATEG() else (constants.getM(), constants.getN())

    conv_last = model_utils.convolutionLayer(final_conv_2,n_chann,(1,1),act_name,l1_l2_reg,kernel_init,'same',kernel_constraint,bias_constraint,is2D=True)
    general_utils.print_int_shape(conv_last)
    y = layers.Reshape(shape_output)(conv_last)
    general_utils.print_int_shape(y)

    model_base = models.Model(vgg16_model.input, y)

    model = models.Model(inputs=inputs, outputs=model_base(conv_3))

    return model


################################################################################
# mJ-Net model with 4D data (3D+time) as input
def mJNet_3dot5D(params, multiInput, usePMs=True, batch=True, drop=False, leaky=True, attentiongate=True):
    size_two = (2,2,1) if constants.getTIMELAST() else (1,2,2)
    kernel_size = (3,3,1) if constants.getTIMELAST() else (1,3,3)
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    activ_func = None if leaky else 'relu'
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN(), 1)
    n_slices = 0 if "n_slices" not in params.keys() else params["n_slices"]
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.getKernelInit(params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["bias_constraint"])

    conv_out, block_6, block_5, block_4, block_3, inputs = [], [], [], [], [], []
    for slice in range(1,n_slices+1):
        input_x = layers.Input(shape=input_shape, sparse=False)
        inputs.append(input_x)

        kernel_shape = (3,3,constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,3,3)
        pool_size = (1,1,params["max_pool"][str(slice)+".long.1"]) if constants.getTIMELAST() else (params["max_pool"][str(slice)+".long.1"],1,1)
        out_1 = model_utils.blockConv3D(input_x,[8,8],kernel_shape,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,pool_size)
        new_z = int(constants.NUMBER_OF_IMAGE_PER_SECTION/params["max_pool"][str(slice)+".long.1"])
        kernel_shape = (3,3,new_z) if constants.getTIMELAST() else (new_z,3,3)
        pool_size = (1,1,params["max_pool"][str(slice)+".long.2"]) if constants.getTIMELAST() else (params["max_pool"][str(slice)+".long.2"],1,1)
        out_2 = model_utils.blockConv3D(out_1,[16,16],kernel_shape,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,pool_size)
        new_z = int(constants.NUMBER_OF_IMAGE_PER_SECTION/params["max_pool"][str(slice)+".long.2"])
        kernel_shape = (3,3,new_z) if constants.getTIMELAST() else (new_z,3,3)
        pool_size = (1,1,params["max_pool"][str(slice)+".long.3"]) if constants.getTIMELAST() else (params["max_pool"][str(slice)+".long.3"],1,1)
        out_3 = model_utils.blockConv3D(out_2,[32,32],kernel_shape,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,pool_size)
        if drop: out_3 = Dropout(params["dropout"][str(slice)+".long.1"])(out_3)
        block_3.append(out_3)
        out_4 = model_utils.blockConv3D(out_3,[8,16],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_4.append(out_4)
        out_5 = model_utils.blockConv3D(out_4,[16,32],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_5.append(out_5)
        out_6 = model_utils.blockConv3D(out_5,[32,64],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        block_6.append(out_6)
        out_7 = model_utils.blockConv3D(out_6,[64,128],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
        if drop: out_7 = Dropout(params["dropout"][str(slice) + ".1"])(out_7)
        conv_out.append(out_7)

    if usePMs:
        layersAfterTransferLearning = []
        PMS = model_utils.getPMsList(multiInput, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)

        for pm in PMS:
            layersAfterTransferLearning.append(pm.conv_2)
            inputs.append(pm.input)
            block5_conv3 = pm.layer_dict["block5_conv3" + pm.name].output
            conv_out.append(layers.Reshape((block5_conv3.shape[1],block5_conv3.shape[2],1,block5_conv3.shape[3]))(block5_conv3))
            block4_conv3 = pm.layer_dict["block4_conv3" + pm.name].output
            block_6.append(layers.Reshape((block4_conv3.shape[1],block4_conv3.shape[2],1,block4_conv3.shape[3]))(block4_conv3))
            block3_conv3 = pm.layer_dict["block3_conv3" + pm.name].output
            block_5.append(layers.Reshape((block3_conv3.shape[1],block3_conv3.shape[2],1,block3_conv3.shape[3]))(block3_conv3))
            block2_conv2 = pm.layer_dict["block2_conv2" + pm.name].output
            block_4.append(layers.Reshape((block2_conv2.shape[1],block2_conv2.shape[2],1,block2_conv2.shape[3]))(block2_conv2))
            block1_conv2 = pm.layer_dict["block1_conv2" + pm.name].output
            block_3.append(layers.Reshape((block1_conv2.shape[1],block1_conv2.shape[2],1,block1_conv2.shape[3]))(block1_conv2))

        conc_layer = layers.Concatenate(-1)(layersAfterTransferLearning)
        transp_1 = layers.Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2), padding='same',activation=activ_func,
                                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_layer)
        conv_out.append(layers.Reshape((transp_1.shape[1],transp_1.shape[2],1,transp_1.shape[3]))(transp_1))

    # check if there is a need to add more info in the input (NIHSS, gender, ...)
    inputs, conv_out, _, _ = model_utils.addMoreInfo(multiInput, inputs, conv_out, [], [], is3D=True, is4D=True)
    if len(conv_out)>1: conv_out = layers.Concatenate(-1)(conv_out)
    elif len(conv_out)==1: conv_out = conv_out[0]
    if attentiongate:
        block_6_conc = layers.Concatenate(-1)(block_6)
        attGate_1 = model_utils.attentionGateBlock(x=block_6_conc, g=conv_out, inter_shape=64, l1_l2_reg=l1_l2_reg, kernel_init=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        up_1 = layers.Concatenate(-1)([layers.UpSampling3D(size=size_two)(conv_out),attGate_1])
        block_5_conc = layers.Concatenate(-1)(block_5)
        up_2 = model_utils.upSamplingPlusAttention(up_1,block_5_conc,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        block_4_conc = layers.Concatenate(-1)(block_4)
        up_3 = model_utils.upSamplingPlusAttention(up_2,block_4_conc,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        block_3_conc = layers.Concatenate(-1)(block_3)
        up_4 = model_utils.upSamplingPlusAttention(up_3,block_3_conc,[8,8,8],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)

    else:
        # TODO: use this part in combination with PMs (?)
        transp_1 = Conv3DTranspose(128, kernel_size=size_two, strides=size_two, activation=activ_func,
                                   padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_out)
        if leaky: transp_1 = layers.LeakyReLU(alpha=0.33)(transp_1)

        block_6_conc = layers.Concatenate(-1)(block_6)
        up_1 = layers.Concatenate(-1)([transp_1, block_6_conc])

        up_2 = model_utils.upLayers(up_1,block_5,[64,64,64],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        up_3 = model_utils.upLayers(up_2,block_4,[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)
        up_4 = model_utils.upLayers(up_3,block_3,[16,16,16],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky)

    # set the softmax activation function if the flag is set
    act_name = "softmax" if constants.getTO_CATEG() else "sigmoid"
    n_chann = len(constants.LABELS) if constants.getTO_CATEG() else 1
    shape_output = (constants.getM(), constants.getN(), n_chann) if constants.getTO_CATEG() else (constants.getM(), constants.getN())

    final_conv = Conv3D(n_chann, kernel_size=(1,1,1), activation=act_name, padding='same', kernel_regularizer=l1_l2_reg,
                        kernel_initializer=kernel_init,kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_4)
    y = layers.Reshape(shape_output)(final_conv)

    model = models.Model(inputs=inputs, outputs=[y])
    return model


################################################################################
# mJ-Net model with 4D data (3D+time) as input
def mJNet_4D(params, multiInput, usePMs=True, batch=True, drop=False, leaky=True, attentiongate=True):
    assert "n_slices" in params.keys(), "Expecting # of slices > 0"

    size_two = (2,2,1) if constants.getTIMELAST() else (1,2,2)
    kernel_size = (3,3,1) if constants.getTIMELAST() else (1,3,3)
    n_slices = params["n_slices"]
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    activ_func = None if leaky else 'relu'
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1) \
        if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN(),1)
    reshape_input_shape = (constants.getM(), constants.getN(), 1, constants.NUMBER_OF_IMAGE_PER_SECTION, 1) \
        if constants.getTIMELAST() else (1,constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN(),1)
    z_axis = 3 if constants.getTIMELAST() else 1
    kernel_init = "glorot_uniform" if "kernel_init" not in params.keys() else model_utils.getKernelInit(params["kernel_init"])
    kernel_constraint = None if "kernel_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["kernel_constraint"])
    bias_constraint = None if "bias_constraint" not in params.keys() else model_utils.getKernelBiasConstraint(params["bias_constraint"])

    # reduce t dimension
    inputs,conc_inputs = [],[]
    for _ in range(n_slices):
        inp = layers.Input(shape=input_shape, sparse=False)
        inputs.append(inp)
        conc_inputs.append(layers.Reshape(reshape_input_shape)(inp))
    conc_inputs = layers.Concatenate(axis=z_axis)(conc_inputs)
    general_utils.print_int_shape(conc_inputs)
    kernel_shape = (3,3,n_slices,3) if constants.getTIMELAST() else (3,n_slices,3,3)
    stride_size = (1,1,params["stride"]["long.1"]) if constants.getTIMELAST() else (params["stride"]["long.1"],1,1)
    out_1 = model_utils.block4DConv(conc_inputs,[8,8,8],kernel_shape,activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                                    bias_constraint, leaky, batch, stride_size)
    general_utils.print_int_shape(out_1)

    kernel_shape = (3,3,n_slices,3) if constants.getTIMELAST() else (3,n_slices,3,3)
    stride_size = (1,1,params["stride"]["long.2"]) if constants.getTIMELAST() else (params["stride"]["long.2"],1,1)
    out_2 = model_utils.block4DConv(out_1,[16,16,16],kernel_shape,activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                                    bias_constraint, leaky, batch, stride_size)
    general_utils.print_int_shape(out_2)
    kernel_shape = (3,3,n_slices,3) if constants.getTIMELAST() else (3,n_slices,3,3)
    stride_size = (1,1,params["stride"]["long.3"]) if constants.getTIMELAST() else (params["stride"]["long.3"],1,1)
    out_3 = model_utils.block4DConv(out_2,[32,32,32],kernel_shape,activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                                    bias_constraint, leaky, batch, stride_size)
    general_utils.print_int_shape(out_3)

    out_shape = (n_slices,constants.getM(),constants.getN(),32)
    out_3 = layers.Reshape(out_shape)(out_3)
    if drop: out_3 = Dropout(params["dropout"]["long.1"])(out_3)
    general_utils.print_int_shape(out_3)

    # reduce z dimension
    out_4 = model_utils.blockConv3D(out_3,[8,16],(n_slices,3,3),activ_func,l1_l2_reg,kernel_init,kernel_constraint,
                                    bias_constraint,leaky,batch,(n_slices,1,1))
    general_utils.print_int_shape(out_4)
    # reduce (x,y) dimension
    out_5 = model_utils.blockConv3D(out_4,[16,32],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
    general_utils.print_int_shape(out_5)
    out_6 = model_utils.blockConv3D(out_5,[32,64],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
    general_utils.print_int_shape(out_6)
    out_7 = model_utils.blockConv3D(out_6,[64,128],kernel_size,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,leaky,batch,size_two)
    general_utils.print_int_shape(out_7)
    if drop: out_7 = Dropout(params["dropout"]["1"])(out_7)

    transp_1 = Conv3DTranspose(128, kernel_size=size_two, strides=size_two, activation=activ_func,
                               padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(out_7)
    if leaky: transp_1 = layers.LeakyReLU(alpha=0.33)(transp_1)
    general_utils.print_int_shape(transp_1)
    up_1 = layers.Concatenate(-1)([transp_1, out_6])
    general_utils.print_int_shape(up_1)
    up_2 = model_utils.upLayers(up_1,[out_5],[64,64,64],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,params,leaky=leaky)
    general_utils.print_int_shape(up_2)
    up_3 = model_utils.upLayers(up_2,[out_4],[32,32,32],kernel_size,size_two,activ_func,l1_l2_reg,kernel_init,kernel_constraint,bias_constraint,params,leaky=leaky)
    general_utils.print_int_shape(up_3)
    # set the softmax activation function if the flag is set
    act_name = "softmax" if constants.getTO_CATEG() else "sigmoid"
    n_chann = len(constants.LABELS) if constants.getTO_CATEG() else 1
    shape_output = (constants.getM(), constants.getN(), n_chann) if constants.getTO_CATEG() else (constants.getM(), constants.getN())

    final_conv = Conv3D(n_chann, kernel_size=(1,1,1), activation=act_name, padding='same', kernel_regularizer=l1_l2_reg,
                        kernel_initializer=kernel_init,kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_3)
    y = layers.Reshape(shape_output)(final_conv)
    general_utils.print_int_shape(y)
    model = models.Model(inputs=inputs, outputs=[y])
    return model