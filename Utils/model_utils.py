import cv2
import glob
import numpy as np
import tensorflow.keras.backend as K
from scipy import ndimage
from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv2D, Conv3D, Concatenate, Conv2DTranspose, Conv3DTranspose, Dropout, \
    TimeDistributed

from Model import constants
from Utils import general_utils, Conv4D


################################################################################
# Class that define a PM object
class PM_obj(object):
    def __init__(self, name, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
        self.name = ("_" + name)
        self.input_shape = (constants.getM(), constants.getN(), 3)
        self.chan = 1 if params["convertImgToGray"] else 3
        self.input, self.input_tensor, self.pre_input, self.pre_model = None, None, None, None

        self.weights = 'imagenet'

        if params["concatenate_input"]:  # concatenate the PMs (RGB or Gray)
            self.pre_input = []
            for pm in constants.getList_PMS():
                if pm.lower() in params["multiInput"].keys() and params["multiInput"][pm.lower()] == 1:
                    inp_shape = (constants.getM(), constants.getN(), self.chan) if not params["inflate_network"] else (1, constants.getM(), constants.getN(), self.chan)
                    self.pre_input.append(layers.Input(shape=inp_shape))
            # concatenate on the end if there is no inflation else concatenate on the first dimension (temporal)
            axis = -1 if not params["inflate_network"] else 1
            self.input_tensor = layers.Concatenate(axis=axis)(self.pre_input)
            # if we don't want to inflate, don't do a conv layer to squeeze the channel dimension
            if not params["inflate_network"]:
                self.input_tensor = Conv2D(3, kernel_size=(3, 3), padding='same', activation=activ_func,
                                           kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.input_tensor)

        # Create base model
        self.base_model = VGG16(weights=self.weights, include_top=False, input_shape=self.input_shape)
        self.base_model._name += self.name

        self.inflateModel = models.Sequential() if params["inflate_network"] else None
        self.dictWeights = dict()

        for layer in self.base_model.layers:
            layer._name += self.name
            if params["inflate_network"] and params["concatenate_input"]: self.inflateVGG16Layer(layer)

        if params["inflate_network"] and params["concatenate_input"]: self.base_model = self.inflateModel
        # Freeze base model
        self.base_model.trainable = False if params["trainable"]==0 else True
        self.input = self.base_model.input
        # Creating dictionary that maps layer names to the layers
        self.layer_dict = dict([(layer.name, layer) for layer in self.base_model.layers])

        # Conv layers after the VGG16
        input_layer = self.base_model.output
        if params["inflate_network"] and params["concatenate_input"]:
            input_layer = layers.Reshape((self.base_model.output.shape[2],self.base_model.output.shape[3],
                                          self.base_model.output.shape[1]*self.base_model.output.shape[-1]))(self.base_model.output)
        self.conv_1 = Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                             kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                             kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_layer)
        self.conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                             kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                             kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.conv_1)
        if batch: self.conv_2 = layers.BatchNormalization()(self.conv_2)
        self.conv_2 = Dropout(params["dropout"][name+".1"])(self.conv_2)

    # inflate the layer for the new VGG16 architecture
    def inflateVGG16Layer(self, layer):
        newLayer = None
        weights, biases = np.empty(0), np.empty(0)
        config = layer.get_config()
        if len(layer.weights) > 0:
            weights, biases = layer.get_weights()
            weights = np.stack([weights/3]*3, axis=0)

        # overwrite the input shape and inflate the conv and pooling layers from 2D to 3D
        if type(layer)==layers.InputLayer: newLayer = layers.Input(shape=(len(self.pre_input), constants.getM(), constants.getN(), self.chan))
        elif type(layer)==layers.Conv2D:
            newLayer = layers.Conv3D(config["filters"], kernel_size=config["kernel_size"][0], strides=config["strides"][0],
                                     padding=config["padding"], dilation_rate=config["dilation_rate"][0],activation=config["activation"])
            self.dictWeights[layer._name] = (weights, biases)
        elif type(layer)==layers.MaxPooling2D:
            recep_field = (1,1) if "block4" not in layer.name and "block5" not in layer.name else (config["pool_size"][0], config["strides"][0])
            padding = config["padding"] if "block4" not in layer.name and "block5" not in layer.name else "same"
            newLayer = layers.MaxPooling3D(pool_size=(recep_field[0],)+config["pool_size"], padding=padding, strides=(recep_field[1],)+config["strides"])
        else: print("we have a problem: ", type(layer), layers.Input, type(layer)==layers.Input)

        # set the name and add the new layer
        if newLayer is not None:
            newLayer._name = layer._name
            self.inflateModel.add(newLayer)


################################################################################
# Get the list of PMs classes
def getPMsList(multiInput, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
    PMS = []
    if params["concatenate_input"]:
        concat = PM_obj("concat", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(concat)
    else:
        if "cbf" in multiInput.keys() and multiInput["cbf"] == 1:
            cbf = PM_obj("cbf", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(cbf)
        if "cbv" in multiInput.keys() and multiInput["cbv"] == 1:
            cbv = PM_obj("cbv", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(cbv)
        if "ttp" in multiInput.keys() and multiInput["ttp"] == 1:
            ttp = PM_obj("ttp", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(ttp)
        if "mtt" in multiInput.keys() and multiInput["mtt"] == 1:
            mtt = PM_obj("mtt", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(mtt)
        if "tmax" in multiInput.keys() and multiInput["tmax"] == 1:
            tmax = PM_obj("tmax", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(tmax)
        if "mip" in multiInput.keys() and multiInput["mip"]==1:
            mip = PM_obj("mip", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(mip)
    return PMS


################################################################################
# Function to get the input X depending on the correct model
def getCorrectXForInputModel(nn, current_folder, row, batchIndex, batch_length, X=None, train=False):
    pms = dict()
    # Extract the information: coordinates, data_aug_idx, ...
    coord = row["x_y"] if train else row["x_y"].iloc[0]
    data_aug_idx = row["data_aug_idx"] if train else row["data_aug_idx"].iloc[0]
    sliceIndex = row["sliceIndex"] if train else row["sliceIndex"].iloc[0]
    # Set the folders with the current one
    folders = [current_folder]
    if (nn.is4DModel or nn.is3dot5DModel) and nn.n_slices > 1: folders = getPrevNextFolder(current_folder, sliceIndex)
    # Get the shape of the input X
    if not train:
        x_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION) if constants.getTIMELAST() else (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN())
        x_shape = (1,)+x_shape+(1,)
        X = np.zeros(shape=x_shape)
    # Important flag. check if X should be an array or not
    isXarray = True if len(folders) > 1 or (nn.x_label == constants.getList_PMS() or (nn.x_label == "pixels" and (nn.is4DModel or nn.is3dot5DModel))) else False
    if isXarray and (not train or (train and batchIndex==0)): X = [None] * len(folders)

    for z, folder in enumerate(folders):
        tmpX = np.empty((batch_length, constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)) if constants.getTIMELAST() else np.empty((batch_length, constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN(), 1))
        if isXarray and train and batchIndex>0: tmpX = X[z]
        howmany = len(glob.glob(folder + "*.*"))
        interpX = np.empty((constants.getM(),constants.getN(),howmany,1)) if constants.getTIMELAST() else np.empty((howmany,constants.getM(),constants.getN(),1))

        for timeIndex, filename in enumerate(np.sort(glob.glob(folder + "*.*"))):
            totimg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            assert totimg is not None, "The image {} is None".format(filename)
            # Get the slice and if we are training, also perform augmentation
            sliceW = general_utils.getSlicingWindow(totimg, coord[0], coord[1])
            if train and not constants.getIsISLES2018(): sliceW = general_utils.performDataAugmentationOnTheImage(sliceW, data_aug_idx)
            if not nn.supervised or nn.patientsFolder != "OLDPREPROC_PATIENTS/":
                # reshape it for the correct input in the model
                if constants.getTIMELAST():
                    if constants.getIsISLES2018(): interpX[:, :, timeIndex, :] = sliceW.reshape(sliceW.shape+(1,)+(1,))  # append the image into a list if ISLES
                    else:
                        if isXarray: tmpX[batchIndex, :, :, timeIndex, :] = sliceW.reshape(sliceW.shape+(1,))
                        else: X[batchIndex, :, :, timeIndex, :] = sliceW.reshape(sliceW.shape+(1,))
                else:
                    if constants.getIsISLES2018(): interpX[timeIndex, :, :, :] = sliceW.reshape((1,)+sliceW.shape+(1,))  # append the image into a list if ISLES
                    else:
                        if isXarray: tmpX[batchIndex, timeIndex, :, :, :] = sliceW.reshape(sliceW.shape+(1,))
                        else: X[batchIndex, timeIndex, :, :, :] = sliceW.reshape(sliceW.shape+(1,))
            else: # here is for the old pre-processing patients (Master 2019)
                if filename != "01.png":
                    if constants.getTIMELAST(): X[:, :, timeIndex] = sliceW
                    else: X[timeIndex, :, :] = sliceW
        ### ISLES2018
        # Interpolation if we are dealing with the ISLES2018 dataset
        if constants.getIsISLES2018():
            axis = -2 if constants.getTIMELAST() else 0
            zoom_val = constants.NUMBER_OF_IMAGE_PER_SECTION/interpX.shape[axis]
            arr_zoom = [1,1,zoom_val,1] if constants.getTIMELAST() else [zoom_val,1,1,1]
            if isXarray: tmpX[batchIndex,:,:,:,:] = ndimage.zoom(interpX, arr_zoom, output=np.float32)
            else: X[batchIndex,:,:,:,:] = ndimage.zoom(interpX, arr_zoom, output=np.float32)

        if isXarray: X[z] = tmpX
        # Check if we are going to add/use the PMs or the additional input (NIHSS, age, gender)
        multiInput = 0
        for k in nn.multiInput.keys(): multiInput+=nn.multiInput[k]

        if multiInput>0:
            if nn.x_label == constants.getList_PMS() or (nn.x_label == "pixels" and (nn.is4DModel or nn.is3dot5DModel)):
                for pm in constants.getList_PMS():
                    if pm not in pms.keys(): pms[pm] = []
                    crn_pm = row[pm] if train else row[pm].iloc[0]
                    totimg = cv2.imread(crn_pm, nn.inputImgFlag)
                    assert totimg is not None, "The image {} is None".format(crn_pm)
                    img = general_utils.getSlicingWindow(totimg, coord[0], coord[1], removeColorBar=True)
                    if train: img = general_utils.performDataAugmentationOnTheImage(img, data_aug_idx)
                    channels = 1 if nn.params["convertImgToGray"] else 3
                    img = np.reshape(img, (constants.getM(), constants.getN(), channels)) if nn.params["convertImgToGray"] else img
                    img = np.reshape(img, (1, constants.getM(), constants.getN(), channels)) if nn.params["inflate_network"] and nn.params["concatenate_input"] else img
                    pms[pm].append(img)

                if "cbf" in nn.multiInput.keys() and nn.multiInput["cbf"] == 1: X.append(np.array(pms["CBF"]))
                if "cbv" in nn.multiInput.keys() and nn.multiInput["cbv"] == 1: X.append(np.array(pms["CBV"]))
                if "ttp" in nn.multiInput.keys() and nn.multiInput["ttp"] == 1: X.append(np.array(pms["TTP"]))
                if "mtt" in nn.multiInput.keys() and nn.multiInput["mtt"] == 1: X.append(np.array(pms["MTT"]))
                if "tmax" in nn.multiInput.keys() and nn.multiInput["tmax"] == 1: X.append(np.array(pms["TMAX"]))
                if "mip" in nn.multiInput.keys() and nn.multiInput["mip"] == 1: X.append(np.array(pms["MIP"]))

            if "nihss" in nn.multiInput.keys() and nn.multiInput["nihss"] == 1:
                nihss_row = row["NIHSS"] if train else row["NIHSS"].iloc[0]
                if nihss_row == "": nihss_row = 0
                X.append(np.array([int(nihss_row)]))
            if "age" in nn.multiInput.keys() and nn.multiInput["age"] == 1:
                age_row = row["age"] if train else row["age"].iloc[0]
                X.append(np.array([int(age_row)]))
            if "gender" in nn.multiInput.keys() and nn.multiInput["gender"] == 1:
                gender_row = row["gender"] if train else row["gender"].iloc[0]
                X.append(np.array([int(gender_row)]))

    return X


################################################################################
# Get the correct regularizer
def getRegularizer(reg_obj):
    regularizer = None
    if reg_obj["type"]=="l1": regularizer = regularizers.l1(l=reg_obj["l"])
    elif reg_obj["type"]=="l2": regularizer = regularizers.l2(l=reg_obj["l"])
    elif reg_obj["type"]=="l1_l2": regularizer = regularizers.l1_l2(l1=reg_obj["l1"], l2=reg_obj["l2"])  # (l1=1e-6, l2=1e-5)
    return regularizer


################################################################################
# Return the correct kernel/bias constraint
def getKernelInit(flag):
    init = "glorot_uniform"
    if flag=="glorot_uniform": init = flag  # Xavier uniform initializer.
    elif flag=="hu_init": init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None) # Hu initializer
    return init


################################################################################
# Return the correct kernel/bias constraint
def getKernelBiasConstraint(flag):
    return max_norm(2.) if flag=="max_norm" else None


################################################################################
# Function to call the convolution layer (2D / 3D)
def convolutionLayer(input, channel, kernel_size, activation, kernel_regularizer, kernel_initializer, padding,
                     kernel_constraint, bias_constraint, strides=1, leaky=False, is2D=False, timedistr=False):
    if is2D: convLayer = Conv2D
    else: convLayer = Conv3D

    if timedistr:  # layer to every temporal slice of an input.
        conv = convLayer(channel, kernel_size=kernel_size, activation=activation, kernel_regularizer=kernel_regularizer, strides=strides,
                         kernel_initializer=kernel_initializer, padding=padding, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        conv = TimeDistributed(conv)(input)
    else:
        conv = convLayer(channel, kernel_size=kernel_size, activation=activation, kernel_regularizer=kernel_regularizer, strides=strides,
                         kernel_initializer=kernel_initializer, padding=padding, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input)
    if leaky: conv = layers.LeakyReLU(alpha=0.33)(conv)
    return conv


################################################################################
# Function to compute two 3D (or 2D) convolutional layers
def doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                      bias_constraint, leaky=False, is2D=False, timedistr=False):
    conv = convolutionLayer(input, channels[0], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=leaky, is2D=is2D, timedistr=timedistr)
    conv = convolutionLayer(conv, channels[1], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=leaky, is2D=is2D, timedistr=timedistr)
    return conv


################################################################################
# Function containing the transpose layers for the deconvolutional part
def upLayers(input, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
             bias_constraint, params, leaky=False, is2D=False):
    if is2D: transposeConv = Conv2DTranspose
    else: transposeConv = Conv3DTranspose

    conv = doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky, is2D=is2D)
    transp = transposeConv(channels[2], kernel_size=kernel_size, strides=strides_size, padding='same',
                           activation=activ_func, kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)
    if leaky: transp = layers.LeakyReLU(alpha=0.33)(transp)

    if (params["concatenate_input"] and not params["inflate_network"]) or len(block)==1: block_conc = block[0]
    else: block_conc = Concatenate(-1)(block)
    return Concatenate(-1)([transp, block_conc])


################################################################################
# Check if the architecture need to have more info in the input
def addMoreInfo(multiInput, inputs, layersForAppending, pre_input, pre_layer, is3D=False, is4D=False):
    # MORE INFO as input = NIHSS score, age, gender
    input_dim = 0
    concat_input = []
    flag_dense = 0

    if "nihss" in multiInput.keys() and multiInput["nihss"] == 1:
        flag_dense = 1
        input_dim += 1
        concat_input.append(layers.Input(shape=(1,)))
    if "age" in multiInput.keys() and multiInput["age"] == 1:
        flag_dense = 1
        input_dim += 1
        concat_input.append(layers.Input(shape=(1,)))
    if "gender" in multiInput.keys() and multiInput["gender"] == 1:
        flag_dense = 1
        input_dim += 1
        concat_input.append(layers.Input(shape=(1,)))

    if flag_dense:
        if input_dim == 1: conc = concat_input[0]
        else: conc = Concatenate(1)(concat_input)
        dense_1 = layers.Dense(100, input_dim=input_dim, activation="relu")(conc)
        third_dim = 1 if not is3D else layersForAppending[0].shape[3]
        fourth_dim = 1 if not is4D else layersForAppending[0].shape[4]

        dense_2 = layers.Dense(layersForAppending[0].shape[1] * layersForAppending[0].shape[2] * third_dim * fourth_dim, activation="relu")(dense_1)
        out = layers.Reshape((layersForAppending[0].shape[1], layersForAppending[0].shape[2], third_dim))(dense_2)
        if is4D: out = layers.Reshape((layersForAppending[0].shape[1], layersForAppending[0].shape[2], third_dim, fourth_dim))(dense_2)
        multiInput_mdl = models.Model(inputs=concat_input, outputs=[out])
        inputs = [inputs, concat_input]
        pre_input = [pre_input, concat_input]
        pre_layer = [pre_layer, concat_input]
        layersForAppending.append(multiInput_mdl.output)

    return inputs, layersForAppending, pre_input, pre_layer


################################################################################
# Function containing a block for the convolutional part
def blockConv3D(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
              leaky, batch, pool_size):
    conv_1 = convolutionLayer(input, channel=channels[0], kernel_size=kernel_size, activation=activ_func, leaky=leaky,
                              kernel_initializer=kernel_init, padding='same', kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint, kernel_regularizer=l1_l2_reg)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    conv_1 = convolutionLayer(conv_1, channel=channels[1], kernel_size=kernel_size, activation=activ_func, leaky=leaky,
                              kernel_initializer=kernel_init, padding='same', kernel_constraint=kernel_constraint,
                              bias_constraint=bias_constraint, kernel_regularizer=l1_l2_reg)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    return layers.MaxPooling3D(pool_size)(conv_1)


################################################################################
# Function to execute a double 3D convolution, followed by an attention gate, upsampling, and concatenation
def upSamplingPlusAttention(input, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init,
                            kernel_constraint, bias_constraint, leaky, is2D=False):
    conv = doubleConvolution(input, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky, is2D)

    attGate = attentionGateBlock(x=block, g=conv, inter_shape=channels[2], l1_l2_reg=l1_l2_reg, kernel_init=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, is2D=is2D)
    if is2D: up = layers.concatenate([layers.UpSampling2D(size=strides_size)(conv), attGate], axis=-1)
    else: up = layers.concatenate([layers.UpSampling3D(size=strides_size)(conv), attGate], axis=-1)
    return up


################################################################################
# Get the previous and next folder, given a specific folder and the slice index
def getPrevNextFolder(folder, sliceIndex):
    folders = []
    maxSlice = len(glob.glob(folder[:-3]+"*"))
    if int(sliceIndex)==1:
        folders.extend([folder,folder])
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)+1)+"/"))
    elif int(sliceIndex)==maxSlice:
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)-1)+"/"))
        folders.extend([folder, folder])
    else:
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)-1)+"/"))
        folders.append(folder)
        folders.append(folder.replace("/"+sliceIndex+"/","/"+general_utils.getStringFromIndex(int(sliceIndex)+1)+"/"))

    return folders


################################################################################
# Anonymous lambda function to expand the specified axis by a factor of argument, rep.
# If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2
def expend_as(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)


################################################################################
# Attention gate block; from: https://arxiv.org/pdf/1804.03999.pdf
def attentionGateBlock(x, g, inter_shape, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=False):
    shape_x = tuple([i for i in list(K.int_shape(x[0])) if i])
    shape_g = tuple([i for i in list(K.int_shape(g[0])) if i])

    if is2D:
        convLayer = Conv2D
        upsampling = layers.UpSampling2D
        strides = (shape_x[0]//shape_g[0],shape_x[1]//shape_g[1])
    else:
        convLayer = Conv3D
        upsampling = layers.UpSampling3D
        strides = (shape_x[0]//shape_g[0],shape_x[1]//shape_g[1],shape_x[2]//shape_g[2])

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = convLayer(inter_shape, kernel_size=1, padding='same', kernel_regularizer=l1_l2_reg,
                      kernel_initializer=kernel_init, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(g)
    # Getting the x signal to the same shape as the gating signal
    theta_x = convLayer(inter_shape, kernel_size=3, padding='same', kernel_regularizer=l1_l2_reg,
                        kernel_initializer=kernel_init, kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint, strides=strides)(x)
    # Element-wise addition of the gating and x signals
    add_xg = layers.Add()([phi_g, theta_x])
    add_xg = layers.Activation('relu')(add_xg)
    # 1x1x1 convolution
    psi = convLayer(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = layers.Activation('sigmoid')(psi)
    shape_sigmoid = tuple([i for i in list(K.int_shape(psi)) if i])
    if is2D: up_size = (shape_x[0]//shape_sigmoid[0],shape_x[1]//shape_sigmoid[1])
    else: up_size = (shape_x[0]//shape_sigmoid[0],shape_x[1]//shape_sigmoid[1],shape_x[2]//shape_sigmoid[2])
    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = upsampling(size=up_size)(psi)
    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[-1])
    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = layers.Multiply()([upsample_sigmoid_xg, x])
    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = convLayer(filters=shape_x[-1], kernel_size=1, padding='same')(attn_coefficients)
    output = layers.BatchNormalization()(output)
    return output


################################################################################
# Squeeze and excite block
def squeeze_excite_block(inputs, ratio=8):
    filters = inputs.shape[-1]
    se_shape = (1,1,filters)

    se = layers.GlobalAveragePooling3D()(inputs)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return layers.Multiply()([inputs, se])


################################################################################
# ResNet block
def resNetBlock(input, k_reg, k_init, k_constraint, bias_constraint, filter, strides):
    conv_1 = layers.BatchNormalization()(input)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_1 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", strides=strides,
                    kernel_initializer=k_init, kernel_constraint=k_constraint, bias_constraint=bias_constraint)(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_1 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint)(conv_1)

    conv_2 = Conv3D(filter, kernel_size=(1, 1, 1), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, strides=strides)(input)
    conv_2 = layers.BatchNormalization()(conv_2)

    add = layers.Add()([conv_1, conv_2])
    return squeeze_excite_block(add)


################################################################################
# Atrous Spatial Pyramidal Pooling block
def ASSP(input, k_reg, k_init, k_constraint, bias_constraint, filter, r_scale=1):
    d_rate = (6*r_scale,6*r_scale,6*r_scale)
    conv_1 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(input)
    conv_1 = layers.BatchNormalization()(conv_1)

    d_rate = (12*r_scale,12*r_scale,12*r_scale)
    conv_2 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(input)
    conv_2 = layers.BatchNormalization()(conv_2)

    d_rate = (18*r_scale,18*r_scale,18*r_scale)
    conv_3 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(input)
    conv_3 = layers.BatchNormalization()(conv_3)

    conv_4 = Conv3D(filter, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint)(input)
    conv_4 = layers.BatchNormalization()(conv_4)

    add = layers.Add()([conv_1, conv_2, conv_3, conv_4])
    return Conv3D(filter, kernel_size=(1,1,1), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                  kernel_constraint=k_constraint, bias_constraint=bias_constraint)(add)


################################################################################
#
def block4DConv(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                leaky, batch, reduce_dim, stride_size):
    conv_1 = Conv4D.Conv4D(inp, channels[0], kernel_size=kernel_size, activation=activ_func, reduce_dim=reduce_dim,
                           kernel_initializer=kernel_init, kernel_regularizer=l1_l2_reg,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    if leaky: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    conv_2 = Conv4D.Conv4D(conv_1, channels[1], kernel_size=kernel_size, activation=activ_func, reduce_dim=reduce_dim,
                           kernel_initializer=kernel_init, kernel_regularizer=l1_l2_reg,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    if leaky: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    if batch: conv_2 = layers.BatchNormalization()(conv_2)

    conv_3 = Conv4D.Conv4D(conv_2, channels[2], kernel_size=kernel_size, strides=stride_size, activation=activ_func,
                           kernel_initializer=kernel_init, kernel_regularizer=l1_l2_reg, reduce_dim=reduce_dim,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    if leaky: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)

    return conv_3
