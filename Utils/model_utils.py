import cv2, glob, platform
import numpy as np
import tensorflow.keras.backend as K
from scipy import ndimage
from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv2D,Conv3D,Concatenate,Conv2DTranspose,Conv3DTranspose,Dropout,TimeDistributed

from Model.constants import *
from Utils import general_utils, layers_4D


################################################################################
# Class that defines a Monte Carlo dropout layer
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


################################################################################
# Class that define a PM object
class PM_obj(object):
    def __init__(self, name, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
        self.name = ("_" + name)
        self.input_shape = (get_m(), get_n(), 3)
        self.chan = 1 if params["convertImgToGray"] else 3
        self.input, self.input_tensor, self.pre_input, self.pre_model = None, None, None, None

        self.weights = 'imagenet'

        if params["concatenate_input"]:  # concatenate the PMs (RGB or Gray)
            self.pre_input = []
            for pm in get_list_PMS():
                if pm.lower() in params["multiInput"].keys() and params["multiInput"][pm.lower()] == 1:
                    inp_shape = (get_m(), get_n(), self.chan) if not params["inflate_network"] else (1, get_m(), get_n(), self.chan)
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
        if type(layer)==layers.InputLayer: newLayer = layers.Input(shape=(len(self.pre_input), get_m(), get_n(), self.chan))
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
def get_PMs_list(multi_input, params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch):
    PMS = []
    if params["concatenate_input"]:
        concat = PM_obj("concat", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
        PMS.append(concat)
    else:
        if "cbf" in multi_input.keys() and multi_input["cbf"] == 1:
            cbf = PM_obj("cbf", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(cbf)
        if "cbv" in multi_input.keys() and multi_input["cbv"] == 1:
            cbv = PM_obj("cbv", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(cbv)
        if "ttp" in multi_input.keys() and multi_input["ttp"] == 1:
            ttp = PM_obj("ttp", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(ttp)
        if "mtt" in multi_input.keys() and multi_input["mtt"] == 1:
            mtt = PM_obj("mtt", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(mtt)
        if "tmax" in multi_input.keys() and multi_input["tmax"] == 1:
            tmax = PM_obj("tmax", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(tmax)
        if "mip" in multi_input.keys() and multi_input["mip"]==1:
            mip = PM_obj("mip", params, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, batch)
            PMS.append(mip)
    return PMS


################################################################################
# Get the initial parameters for 2D models
def get_init_params_2D(params,leaky):
    out_params = {
        "size_two": (2,2),
        "kernel_size": (3,3),
        "l1_l2_reg": None if "regularizer" not in params.keys() else get_regularizer(params["regularizer"]),
        "activ_func": None if leaky else 'relu',
        "input_shape": (get_m(), get_n(), 1),
        "n_slices": 0 if "n_slices" not in params.keys() else params["n_slices"],
        "kernel_init": "glorot_uniform" if "kernel_init" not in params.keys() else get_kernel_init(params["kernel_init"]),
        "kernel_constraint": None if "kernel_constraint" not in params.keys() else get_kernel_bias_constraint(params["kernel_constraint"]),
        "bias_constraint": None if "bias_constraint" not in params.keys() else get_kernel_bias_constraint(params["bias_constraint"]),
        "reduce_dim": params["reduce_dim"] if "reduce_dim" in params.keys() else 2,
        "reshape_input_shape": (get_m(), get_n(), 1, getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (1, getNUMBER_OF_IMAGE_PER_SECTION(), get_m(), get_n(), 1),
        "z_axis": 3 if is_timelast() else 1
    }
    out_params["input_shape_3D"] = (get_m(), get_n(), out_params["n_slices"], 1) if is_timelast() else (out_params["n_slices"], get_m(), get_n(), 1),
    return out_params


################################################################################
# Get the initial parameters for 3D/4D models
def get_init_params_3D(params,leaky):
    out_params = {
        "size_two": (2,2,1) if is_timelast() else (1,2,2),
        "kernel_size": (3,3,1) if is_timelast() else (1,3,3),
        "l1_l2_reg": None if "regularizer" not in params.keys() else get_regularizer(params["regularizer"]),
        "activ_func": None if leaky else 'relu',
        "input_shape": (get_m(), get_n(), getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(), get_m(), get_n(), 1),
        "n_slices": 0 if "n_slices" not in params.keys() else params["n_slices"],
        "kernel_init": "glorot_uniform" if "kernel_init" not in params.keys() else get_kernel_init(params["kernel_init"]),
        "kernel_constraint": None if "kernel_constraint" not in params.keys() else get_kernel_bias_constraint(params["kernel_constraint"]),
        "bias_constraint": None if "bias_constraint" not in params.keys() else get_kernel_bias_constraint(params["bias_constraint"]),
        "reduce_dim": params["reduce_dim"] if "reduce_dim" in params.keys() else 2,
        "reshape_input_shape": (get_m(), get_n(), 1, getNUMBER_OF_IMAGE_PER_SECTION(), 1) if is_timelast() else (1, getNUMBER_OF_IMAGE_PER_SECTION(), get_m(), get_n(), 1),
        "z_axis": 3 if is_timelast() else 1
    }
    return out_params


################################################################################
# Function to get the input X depending on the correct model
def get_correct_X_for_input_model(ds_seq, current_folder, row, batch_idx, batch_len, X=None, train=False):
    pms = dict()
    # Extract the information: coordinates, data_aug_idx, ...
    coord = row["x_y"] if train else row["x_y"].iloc[0]
    data_aug_idx = row["data_aug_idx"] if train else row["data_aug_idx"].iloc[0]
    slice_idx = row["sliceIndex"] if train else row["sliceIndex"].iloc[0]
    # Set the folders list with the current one
    folders = [current_folder]
    # Ger the right folders and add them in the list
    if (ds_seq.is4DModel or ds_seq.is3dot5DModel) and ds_seq.n_slices > 1: folders = get_prev_next_folder(current_folder, slice_idx)
    # Get the shape of the input X
    if not train:
        x_shape = (ds_seq.constants["M"], ds_seq.constants["N"], ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"]) if ds_seq.constants["TIME_LAST"] else (ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"], ds_seq.constants["M"], ds_seq.constants["N"])
        X = np.zeros(shape=(1,)+x_shape+(1,))
    # Important flag. Check if the input X should be an array or not
    isXarray = True if len(folders) > 1 or (ds_seq.x_label == ds_seq.constants["LIST_PMS"] or (ds_seq.x_label == "pixels" and (ds_seq.is4DModel or ds_seq.is3dot5DModel))) else False
    isXarray = False if "TCNet" in ds_seq.name else isXarray
    if not train or (train and batch_idx == 0):  # create a list of empty spots: [None,None,...]
        if isXarray: X = [None] * len(folders)
        if "TCNet" in ds_seq.name: X = [None] * ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"]

    for z, folder in enumerate(folders):
        tmp_X = np.empty((batch_len, ds_seq.constants["M"], ds_seq.constants["N"], ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"], 1)) if ds_seq.constants["TIME_LAST"] else np.empty((batch_len, ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"], ds_seq.constants["M"], ds_seq.constants["N"], 1))
        if isXarray and train and batch_idx>0: tmp_X = X[z]

        if platform.system()=="Windows": folder = folder.replace(folder[:folder.rfind("/",0,len(folder)-4)], ds_seq.patients_folder)

        howmany = len(glob.glob(folder + "*.*"))
        interpX = np.empty((ds_seq.constants["M"], ds_seq.constants["N"], howmany, 1)) if ds_seq.constants["TIME_LAST"] else np.empty((howmany, ds_seq.constants["M"], ds_seq.constants["N"], 1))

        # initialize single_X if batch_idx==0, otherwise single_X = X[time_idx]
        single_X = np.empty((batch_len, ds_seq.constants["M"], ds_seq.constants["N"], 1))
        if "TCNet" in ds_seq.name:
            if ds_seq.is3dot5DModel and batch_idx==0 and z==0:
                single_X = np.empty((batch_len, len(folders), ds_seq.constants["M"], ds_seq.constants["N"], 1))

        for time_idx, filename in enumerate(np.sort(glob.glob(folder + "*.*"))):
            if batch_idx>0 or z>0: single_X = X[time_idx]
            totimg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            assert totimg is not None, "The image {} is None".format(filename)
            # Get the slice and if we are training, also perform augmentation
            slc_w = general_utils.get_slice_window(totimg, coord[0], coord[1], ds_seq.constants)
            if train and not ds_seq.constants["isISLES"]: slc_w = general_utils.perform_DA_on_img(slc_w, data_aug_idx)
            if not ds_seq.supervised or ds_seq.patients_folder != "OLDPREPROC_PATIENTS/":
                # reshape it for the correct input in the model
                if ds_seq.constants["TIME_LAST"]:
                    if not ds_seq.constants["isISLES"]:
                        if isXarray:  # 3.5D / 4D
                            tmp_X[batch_idx, :, :, time_idx, :] = slc_w.reshape(slc_w.shape + (1,))
                        elif "TCNet" in ds_seq.name:  # TCN input
                            if ds_seq.is3dot5DModel: single_X[batch_idx, z, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                            else: single_X[batch_idx, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                        else:  # mJNet input (2.5D)
                            X[batch_idx, :, :, time_idx, :] = slc_w.reshape(slc_w.shape + (1,))
                    else:  # append the image into a list if ISLES
                        interpX[:, :, time_idx, :] = slc_w.reshape(slc_w.shape + (1,) + (1,))
                else:
                    if not ds_seq.constants["isISLES"]:
                        if isXarray:  # 3.5D / 4D
                            tmp_X[batch_idx, time_idx, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                        elif "TCNet" in ds_seq.name:  # TCN input
                            if ds_seq.is3dot5DModel: single_X[batch_idx, z, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                            else: single_X[batch_idx, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                        else:  # mJNet input (2.5D)
                            X[batch_idx, time_idx, :, :, :] = slc_w.reshape(slc_w.shape + (1,))
                    else:  # append the image into a list if ISLES
                        interpX[time_idx, :, :, :] = slc_w.reshape((1,) + slc_w.shape + (1,))
            else:  # here is for the old pre-processing patients (Master 2019)
                if filename != "01.png":
                    if ds_seq.constants["TIME_LAST"]: X[:, :, time_idx] = slc_w
                    else: X[time_idx, :, :] = slc_w

            if "TCNet" in ds_seq.name: X[time_idx] = single_X
        # ISLES2018
        # Interpolation if we are dealing with the ISLES2018 dataset
        if ds_seq.constants["isISLES"]:
            axis = -2 if ds_seq.constants["TIME_LAST"] else 0
            zoom_val = ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"] / interpX.shape[axis]
            arr_zoom = [1,1,zoom_val,1] if ds_seq.constants["TIME_LAST"] else [zoom_val, 1, 1, 1]
            if isXarray:tmp_X[batch_idx, :, :, :, :] = ndimage.zoom(interpX, arr_zoom, output=np.float32)
            else:X[batch_idx, :, :, :, :] = ndimage.zoom(interpX, arr_zoom, output=np.float32)

        if isXarray: X[z] = tmp_X
        # Check if we are going to add/use the PMs or the additional input (NIHSS, age, gender)
        are_multi_input = 0
        for k in ds_seq.multi_input.keys(): are_multi_input+=ds_seq.multi_input[k]

        if are_multi_input>0:
            if ds_seq.x_label == ds_seq.constants["LIST_PMS"] or (ds_seq.x_label == "pixels" and (ds_seq.is4DModel or ds_seq.is3dot5DModel)):
                for pm in ds_seq.constants["LIST_PMS"]:
                    if pm not in pms.keys(): pms[pm] = []
                    crn_pm = row[pm] if train else row[pm].iloc[0]
                    totimg = cv2.imread(crn_pm, ds_seq.input_img_flag)
                    assert totimg is not None, "The image {} is None".format(crn_pm)
                    img = general_utils.get_slice_window(totimg, coord[0], coord[1], ds_seq.constants, remove_colorbar=True)
                    if train: img = general_utils.perform_DA_on_img(img, data_aug_idx)
                    channels = 1 if ds_seq.params["convertImgToGray"] else 3
                    img = np.reshape(img, (ds_seq.constants["M"], ds_seq.constants["N"], channels)) if ds_seq.params["convertImgToGray"] else img
                    img = np.reshape(img, (1, ds_seq.constants["M"], ds_seq.constants["N"], channels)) if ds_seq.params["inflate_network"] and ds_seq.params["concatenate_input"] else img
                    pms[pm].append(img)

                if "cbf" in ds_seq.multi_input.keys() and ds_seq.multi_input["cbf"] == 1: X.append(np.array(pms["CBF"]))
                if "cbv" in ds_seq.multi_input.keys() and ds_seq.multi_input["cbv"] == 1: X.append(np.array(pms["CBV"]))
                if "ttp" in ds_seq.multi_input.keys() and ds_seq.multi_input["ttp"] == 1: X.append(np.array(pms["TTP"]))
                if "mtt" in ds_seq.multi_input.keys() and ds_seq.multi_input["mtt"] == 1: X.append(np.array(pms["MTT"]))
                if "tmax" in ds_seq.multi_input.keys() and ds_seq.multi_input["tmax"] == 1: X.append(np.array(pms["TMAX"]))
                if "mip" in ds_seq.multi_input.keys() and ds_seq.multi_input["mip"] == 1: X.append(np.array(pms["MIP"]))

            if "nihss" in ds_seq.multi_input.keys() and ds_seq.multi_input["nihss"] == 1:
                nihss_row = row["NIHSS"] if train else row["NIHSS"].iloc[0]
                if nihss_row == "": nihss_row = 0
                X.append(np.array([int(nihss_row)]))
            if "age" in ds_seq.multi_input.keys() and ds_seq.multi_input["age"] == 1:
                age_row = row["age"] if train else row["age"].iloc[0]
                X.append(np.array([int(age_row)]))
            if "gender" in ds_seq.multi_input.keys() and ds_seq.multi_input["gender"] == 1:
                gender_row = row["gender"] if train else row["gender"].iloc[0]
                X.append(np.array([int(gender_row)]))

    if "TCNet" in ds_seq.name: assert len(X)==ds_seq.constants["NUMBER_OF_IMAGE_PER_SECTION"], "Input does not contain the right amount of TIMEPOINTS"
    if isXarray: assert len(X)==len(folders), "Input does not contain the right amount of FOLDERS"
    return X


################################################################################
# Get the correct regularizer
def get_regularizer(reg_obj):
    regularizer = None
    if reg_obj["type"]=="l1": regularizer = regularizers.l1(l=reg_obj["l"])
    elif reg_obj["type"]=="l2": regularizer = regularizers.l2(l=reg_obj["l"])
    elif reg_obj["type"]=="l1_l2": regularizer = regularizers.l1_l2(l1=reg_obj["l1"], l2=reg_obj["l2"])
    return regularizer


################################################################################
# Return the correct kernel/bias constraint
def get_kernel_init(flag):
    init = "glorot_uniform"  # Xavier uniform initializer.
    if flag=="hu_init": init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None) # Hu initializer
    return init


################################################################################
# Return the correct kernel/bias constraint
def get_kernel_bias_constraint(flag): return max_norm(2.) if flag=="max_norm" else None


################################################################################
# Function to call the convolution layer (2D / 3D)
def convolution_layer(inp, channel, kernel_size, activation, kernel_regularizer, kernel_initializer, padding,
                      kernel_constraint, bias_constraint, strides=1, leaky=False, is2D=False, timedistr=False):
    if is2D: convLayer = Conv2D
    else: convLayer = Conv3D

    if timedistr:  # layer to every temporal slice of an input.
        conv = convLayer(channel, kernel_size=kernel_size, activation=activation, kernel_regularizer=kernel_regularizer,
                         strides=strides, kernel_initializer=kernel_initializer, padding=padding,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        conv = TimeDistributed(conv)(inp)
    else:
        conv = convLayer(channel, kernel_size=kernel_size, activation=activation, kernel_regularizer=kernel_regularizer,
                         strides=strides, kernel_initializer=kernel_initializer, padding=padding,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(inp)
    if leaky: conv = layers.LeakyReLU(alpha=0.33)(conv)
    return conv


################################################################################
# Function to compute two 3D (or 2D) convolutional layers
def double_conv(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                leaky=False, is2D=False, timedistr=False, batch=False):
    conv = convolution_layer(inp, channels[0], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                             kernel_constraint, bias_constraint, leaky=leaky, is2D=is2D, timedistr=timedistr)
    if batch: conv = layers.BatchNormalization()(conv)
    conv = convolution_layer(conv, channels[1], kernel_size, activ_func, l1_l2_reg, kernel_init, 'same',
                             kernel_constraint, bias_constraint, leaky=leaky, is2D=is2D, timedistr=timedistr)
    if batch: conv = layers.BatchNormalization()(conv)
    return conv


################################################################################
# Function containing the transpose layers for the deconvolutional part
def up_layers(inp, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
              bias_constraint, params, leaky=False, is2D=False, batch=False):
    if is2D: transposeConv = Conv2DTranspose
    else: transposeConv = Conv3DTranspose

    conv = double_conv(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint,
                       bias_constraint, leaky, is2D=is2D)
    transp = transposeConv(channels[2], kernel_size=kernel_size, strides=strides_size, padding='same',
                           activation=activ_func, kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)
    if leaky: transp = layers.LeakyReLU(alpha=0.33)(transp)
    if batch: transp = layers.BatchNormalization()(transp)

    if (params["concatenate_input"] and not params["inflate_network"]) or len(block)==1: block_conc = block[0]
    else: block_conc = Concatenate(-1)(block)
    return Concatenate(-1)([transp, block_conc])


################################################################################
# Check if the architecture need to have more info in the input
def add_more_info(multi_input, inputs, append_layers=None, pre_input=None, pre_layer=None, is3D=False, is4D=False):
    # MORE INFO as input = NIHSS score, age, gender
    if pre_layer is None: pre_layer = []
    if pre_input is None: pre_input = []
    if append_layers is None: append_layers = []
    input_dim, flag_dense, concat_input = 0, 0, []

    for key in ["nihss", "age", "gender"]:
        if key in multi_input.keys() and multi_input[key] == 1:
            flag_dense = 1
            input_dim += 1
            concat_input.append(layers.Input(shape=(1,)))

    if flag_dense:
        if input_dim == 1: conc = concat_input[0]
        else: conc = Concatenate(1)(concat_input)
        dense_1 = layers.Dense(100, input_dim=input_dim, activation="relu")(conc)
        third_dim = 1 if not is3D else append_layers[0].shape[3]
        fourth_dim = 1 if not is4D else append_layers[0].shape[4]

        dense_2 = layers.Dense(append_layers[0].shape[1] * append_layers[0].shape[2] * third_dim * fourth_dim, activation="relu")(dense_1)
        out = layers.Reshape((append_layers[0].shape[1], append_layers[0].shape[2], third_dim))(dense_2)
        if is4D: out = layers.Reshape((append_layers[0].shape[1], append_layers[0].shape[2], third_dim, fourth_dim))(dense_2)
        multi_input_mdl = models.Model(inputs=concat_input, outputs=[out])
        inputs = [inputs, concat_input]
        pre_input = [pre_input, concat_input]
        pre_layer = [pre_layer, concat_input]
        append_layers.append(multi_input_mdl.output)

    return inputs, append_layers, pre_input, pre_layer


################################################################################
# Function containing a block for the convolutional part
def block_conv3D(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                 leaky, batch, pool_size):
    conv_1 = convolution_layer(inp, channel=channels[0], kernel_size=kernel_size, activation=activ_func,
                               kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, leaky=leaky)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    conv_1 = convolution_layer(conv_1, channel=channels[1], kernel_size=kernel_size, activation=activ_func,
                               kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init, padding='same',
                               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, leaky=leaky)
    if batch: conv_1 = layers.BatchNormalization()(conv_1)

    return layers.MaxPooling3D(pool_size)(conv_1)


################################################################################
# Function to execute a double 3D convolution, followed by an attention gate, upsampling, and concatenation
def upSamplingPlusAttention(inp, block, channels, kernel_size, strides_size, activ_func, l1_l2_reg, kernel_init,
                            kernel_constraint, bias_constraint, leaky, is2D=False):
    conv = double_conv(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, leaky, is2D)

    attGate = block_attentionGate(x=block, g=conv, inter_shape=channels[2], l1_l2_reg=l1_l2_reg,
                                  kernel_init=kernel_init, kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint, is2D=is2D)
    if is2D: up = layers.concatenate([layers.UpSampling2D(size=strides_size)(conv), attGate], axis=-1)
    else: up = layers.concatenate([layers.UpSampling3D(size=strides_size)(conv), attGate], axis=-1)
    return up


################################################################################
# Get the previous and next folder, given a specific folder and the slice index
def get_prev_next_folder(folder, slice_idx):
    folders = []
    maxSlice = len(glob.glob(folder[:-3]+"*"))
    if int(slice_idx)==1:
        folders.extend([folder,folder])
        folders.append(folder.replace("/"+slice_idx+"/","/"+general_utils.get_str_from_idx(int(slice_idx)+1)+"/"))
    elif int(slice_idx)==maxSlice:
        folders.append(folder.replace("/"+slice_idx+"/","/"+general_utils.get_str_from_idx(int(slice_idx)-1)+"/"))
        folders.extend([folder, folder])
    else:
        folders.append(folder.replace("/"+slice_idx+"/","/"+general_utils.get_str_from_idx(int(slice_idx)-1)+"/"))
        folders.append(folder)
        folders.append(folder.replace("/"+slice_idx+"/","/"+general_utils.get_str_from_idx(int(slice_idx)+1)+"/"))

    return folders


################################################################################
# Anonymous lambda function to expand the specified axis by a factor of argument, rep.
# If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2
def expend_as(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1), arguments={'repnum': rep})(tensor)


################################################################################
# Attention gate block; from: https://arxiv.org/pdf/1804.03999.pdf
def block_attentionGate(x, g, inter_shape, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint, is2D=False):
    shape_x = tuple([i for i in list(K.int_shape(x[0])) if i])
    shape_g = tuple([i for i in list(K.int_shape(g[0])) if i])

    if is2D:
        convLayer = Conv2D
        upsampling = layers.UpSampling2D
        strides = (shape_x[0]//shape_g[0],shape_x[1]//shape_g[1])
    else:  # we are in a 3D situation
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
def block_resNet(inp, k_reg, k_init, k_constraint, bias_constraint, filters, strides):
    conv_1 = layers.BatchNormalization()(inp)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_1 = Conv3D(filters, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", strides=strides,
                    kernel_initializer=k_init, kernel_constraint=k_constraint, bias_constraint=bias_constraint)(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_1 = Conv3D(filters, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint)(conv_1)

    conv_2 = Conv3D(filters, kernel_size=(1, 1, 1), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, strides=strides)(inp)
    conv_2 = layers.BatchNormalization()(conv_2)

    add = layers.Add()([conv_1, conv_2])
    return squeeze_excite_block(add)


################################################################################
# Atrous Spatial Pyramidal Pooling block
def ASSP(inp, k_reg, k_init, k_constraint, bias_constraint, filt, r_scale=1):
    d_rate = (6*r_scale,6*r_scale,6*r_scale)
    conv_1 = Conv3D(filt, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(inp)
    conv_1 = layers.BatchNormalization()(conv_1)

    d_rate = (12*r_scale,12*r_scale,12*r_scale)
    conv_2 = Conv3D(filt, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(inp)
    conv_2 = layers.BatchNormalization()(conv_2)

    d_rate = (18*r_scale,18*r_scale,18*r_scale)
    conv_3 = Conv3D(filt, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint, dilation_rate=d_rate)(inp)
    conv_3 = layers.BatchNormalization()(conv_3)

    conv_4 = Conv3D(filt, kernel_size=(3, 3, 3), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                    kernel_constraint=k_constraint, bias_constraint=bias_constraint)(inp)
    conv_4 = layers.BatchNormalization()(conv_4)

    add = layers.Add()([conv_1, conv_2, conv_3, conv_4])
    return Conv3D(filt, kernel_size=(1, 1, 1), kernel_regularizer=k_reg, padding="same", kernel_initializer=k_init,
                  kernel_constraint=k_constraint, bias_constraint=bias_constraint)(add)


################################################################################
# Block of 4D convolution
def block_conv4D(inp, channels, kernel_size, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                 leaky, batch, reduce_dim, stride_size):

    for i in range(len(channels)-1):
        conv = layers_4D.Conv4D(inp, channels[i], kernel_size=kernel_size, activation=activ_func, reduce_dim=reduce_dim,
                                kernel_initializer=kernel_init, kernel_regularizer=l1_l2_reg,
                                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
        if leaky: conv = layers.LeakyReLU(alpha=0.33)(conv)
        if batch: conv = layers.BatchNormalization()(conv)
        inp = conv

    conv_3 = layers_4D.Conv4D(inp, channels[-1], kernel_size=kernel_size, strides=stride_size, activation=activ_func,
                              kernel_initializer=kernel_init, kernel_regularizer=l1_l2_reg, reduce_dim=reduce_dim,
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    if leaky: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    if batch: conv_3 = layers.BatchNormalization()(conv_3)

    return conv_3, inp


################################################################################
# Blocks for time reduction
def reduce_time_and_space_dim(inputs, variables, params, channels, leaky, batch, drop, last_block=False):
    t_chann = channels
    blocks = []
    inp = inputs[0]
    # reduce time
    for t in range(1,len(inputs)-1):
        key = "long."+str(t)
        stride_size = (1,1,params["stride"][key]) if is_timelast() else (params["stride"][key], 1, 1)
        t_chann = [int(c*2) for c in t_chann]

        block,_ = block_conv4D(inp, t_chann, variables["kernel_shape"], variables["activ_func"], variables["l1_l2_reg"],
                               variables["kernel_init"], variables["kernel_constraint"], variables["bias_constraint"],
                               leaky, batch, 2, stride_size)
        general_utils.print_int_shape(block)
        print("block"+str(t))

        if inputs[t] is not None:  # there is something from the previous layers
            up_chann = [int(c/4) for c in channels]
            up_1 = layers_4D.Conv4DTranspose(inputs[t], up_chann[0],
                                             kernel_size=variables["kernel_shape"],
                                             strides=variables["size_two"],
                                             activation=variables["activ_func"],
                                             kernel_regularizer=variables["l1_l2_reg"],
                                             kernel_initializer=variables["kernel_init"],
                                             kernel_constraint=variables["kernel_constraint"],
                                             bias_constraint=variables["bias_constraint"])
            general_utils.print_int_shape(up_1)
            up_1 = layers.Concatenate(-1)([up_1, block])
            block = layers_4D.Conv4D(up_1,up_chann[0],kernel_size=variables["kernel_shape"],reduce_dim=2,
                                     activation=variables["activ_func"],kernel_initializer=variables["kernel_init"],
                                     kernel_regularizer=variables["l1_l2_reg"],kernel_constraint=variables["kernel_constraint"],
                                     bias_constraint=variables["bias_constraint"])
            if leaky: block = layers.LeakyReLU(alpha=0.33)(block)
            if batch: block = layers.BatchNormalization()(block)

        blocks.append(block)
        inp = block

    # reshape and reduce z dimension
    out_shape = (variables["n_slices"], K.int_shape(inp)[-3], K.int_shape(inp)[-2], K.int_shape(inp)[-1])
    block_z = layers.Reshape(out_shape)(blocks[-1])
    if drop: block_z = Dropout(params["dropout"]["long.1"])(block_z)
    general_utils.print_int_shape(block_z)
    print("block_z - reshape")

    if inputs[-1] is not None:
        block_z_1 = layers.Reshape(out_shape)(inputs[-1])
        if drop: block_z_1 = Dropout(params["dropout"]["long.1"])(block_z_1)
        block_z = layers.Concatenate(-1)([block_z, block_z_1])
    up = block_conv3D(block_z, [K.int_shape(inp)[-1], K.int_shape(inp)[-1]], (variables["n_slices"], 3, 3),
                      variables["activ_func"], variables["l1_l2_reg"], variables["kernel_init"],
                      variables["kernel_constraint"], variables["bias_constraint"], leaky, batch,
                      (variables["n_slices"], 1, 1))
    if last_block:  # if we are in the last block, we don't need to transpose
        blocks.append(up)
        general_utils.print_int_shape(up)
    else:
        up = Conv3DTranspose(K.int_shape(inp)[-1],
                             kernel_size=variables["size_two"],
                             strides=variables["size_two"],
                             activation=variables["activ_func"],
                             padding='same',
                             kernel_regularizer=variables["l1_l2_reg"],
                             kernel_initializer=variables["kernel_init"],
                             kernel_constraint=variables["kernel_constraint"],
                             bias_constraint=variables["bias_constraint"])(up)
        blocks.append(up)
        general_utils.print_int_shape(up)
        print("up")

    return blocks
