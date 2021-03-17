from Model import constants
from Utils import general_utils, spatial_pyramid, model_utils

from tensorflow.keras import layers, models, initializers
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K


################################################################################
# mJ-Net model version for the parametric maps as input
def PMs_segmentation(params, to_categ, multiInput, batch=True):
    activ_func = 'relu'
    l1_l2_reg = None if "regularizer" not in params.keys() else model_utils.getRegularizer(params["regularizer"])
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)
    layersAfterTransferLearning, inputs, block5_conv3, block4_conv3, block3_conv3, block2_conv2, block1_conv2 = [], [], [], [], [], [], []

    PMS = model_utils.getPMsList(multiInput, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)

    for pm in PMS:
        layersAfterTransferLearning.append(pm.conv_2)
        inputs.append(pm.input)
        block5_conv3.append(pm.layer_dict["block5_conv3" + pm.name].output)
        block4_conv3.append(pm.layer_dict["block4_conv3" + pm.name].output)
        block3_conv3.append(pm.layer_dict["block3_conv3" + pm.name].output)
        block2_conv2.append(pm.layer_dict["block2_conv2" + pm.name].output)
        block1_conv2.append(pm.layer_dict["block1_conv2" + pm.name].output)

    # check if there is a need to add more info in the input (NIHSS, gender, ...)
    inputs, layersAfterTransferLearning = model_utils.addMoreInfo(multiInput, inputs, layersAfterTransferLearning)

    conc_layer = layers.Concatenate(-1)(layersAfterTransferLearning)

    transp_1 = layers.Conv2DTranspose(256, kernel_size=(2,2), strides=(2,2), padding='same',activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_layer)

    block5_conv3_conc = layers.Concatenate(-1)(block5_conv3)
    up_1 = layers.Concatenate(-1)([transp_1,block5_conv3_conc])

    # going up with the layers
    up_2 = model_utils.upLayers(up_1, block4_conv3, [128*len(PMS),128*len(PMS),128*len(PMS)], (3,3), (2,2), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=True)
    up_3 = model_utils.upLayers(up_2, block3_conv3, [64*len(PMS),64*len(PMS),64*len(PMS)], (3,3), (2,2), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=True)
    up_4 = model_utils.upLayers(up_3, block2_conv2, [32*len(PMS),32*len(PMS),32*len(PMS)], (3,3), (2,2), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=True)
    up_5 = model_utils.upLayers(up_4, block1_conv2, [16*len(PMS),16*len(PMS),16*len(PMS)], (3,3), (2,2), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=True)

    final_conv_1 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_5)
    if batch: final_conv_1 = layers.BatchNormalization()(final_conv_1)
    # general_utils.print_int_shape(final_conv_1)
    final_conv_2 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_1)
    if batch: final_conv_2 = layers.BatchNormalization()(final_conv_2)
    # general_utils.print_int_shape(final_conv_2)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    final_conv_3 = layers.Conv2D(n_chann, kernel_size=(1, 1), activation=act_name, padding='same',
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_2)
    # general_utils.print_int_shape(final_conv_3)

    y = layers.Reshape(shape_output)(final_conv_3)
    # general_utils.print_int_shape(y)

    model = models.Model(inputs=inputs, outputs=[y])
    return model
