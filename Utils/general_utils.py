# DO NOT import dataset_utils here!
import functools
import warnings

from Model.constants import *
from Utils import metrics, losses

import argparse, os, json, pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
######################## UTILS FUNCTIONS #######################################
# The file should only contains functions!
################################################################################


################################################################################
# get the arguments from the command line
def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("--isles2018", help="Flag to use the ISLES2018 dataset", action="store_true")
    parser.add_argument("-d", "--debug", help="DEBUG mode", action="store_true")
    parser.add_argument("-o", "--original", help="ORIGINAL_SHAPE flag", action="store_true")
    parser.add_argument("-pm", "--pm", help="Use parametric maps", action="store_true")
    parser.add_argument("-t", "--tile", help="Set the tile pixels dimension (MxM)", type=int)
    parser.add_argument("-dim", "--dimension", help="Set the dimension of the input images (widthXheight)", type=int)
    parser.add_argument("-c", "--classes", help="Set the # of classes involved (default = 4)", default=4, type=int, choices=[2,3,4])
    parser.add_argument("-w", "--weights", help="Set the weights for the categorical losses", type=float, nargs='+')
    parser.add_argument("-e", "--exp", help="Set the number of the experiment", type=float)
    parser.add_argument("-j", "--jump", help="Jump the training and go directly on the gradual fine-tuning function", action="store_true")
    parser.add_argument("--timelast", help="Set the time dimension in the last channel of the input model", action="store_true")
    parser.add_argument("--prefix", help="Set the prefix different from the default", type=str)
    parser.add_argument("--limcols", help="Set the columns without additional info", action="store_true")
    parser.add_argument("gpu", help="Give the id of gpu (or a list of the gpus) to use")
    parser.add_argument("sname", help="Select the setting filename")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    set_orig_shape(args.original)
    set_ISLES2018(args.isles2018)
    set_USE_PM(args.pm)
    set_tile_dim(args.tile)
    set_img_dim(args.dimension)
    set_classes(args.classes)
    set_weights(args.weights)
    set_timelast(args.timelast)
    set_prefix(args.prefix)
    set_limited_columns(args.limcols)

    return args


################################################################################
# get the setting file
def get_setting_file(filename):
    # the path of the setting file start from the main.py
    # (= current working directory)
    with open(os.path.join(os.getcwd(), filename)) as f: setting = json.load(f)

    if is_verbose():
        print_sep("-", 50)
        print("Load setting file: {}".format(filename))

    return setting


################################################################################
# setup the global environment
def setup_env(args, setting):
    # important: set up the root path for later uses
    set_rootpath(setting["root_path"])

    if "NUMBER_OF_IMAGE_PER_SECTION" in setting["init"].keys(): setImagePerSection(setting["init"]["NUMBER_OF_IMAGE_PER_SECTION"])
    else: setImagePerSection(30)
    set_3D_flag(True) if "3D" in setting["init"].keys() and setting["init"]["3D"] else set_3D_flag(False)
    if "ONE_TIME_POINT" in setting["init"].keys() and setting["init"]["ONE_TIME_POINT"]: set_onetimepoint(
        get_str_from_idx(
            setting["init"]["ONE_TIME_POINT"]))

    experimentFolder = "EXP" + convert_expnum_to_str(setting["EXPERIMENT"]) + os.path.sep
    N_GPU = setupEnvironmentForGPUs(args, setting)

    for key, rel_path in setting["relative_paths"].items():
        if isinstance(rel_path, dict):
            prefix = key.upper()+os.path.sep
            create_dir(prefix)
            create_dir(prefix + experimentFolder)
            for sub_path in setting["relative_paths"][key].values():
                create_dir(prefix + experimentFolder + sub_path)
        else:
            if rel_path!="": create_dir(rel_path)

    return N_GPU


################################################################################
# setup the environment for the GPUs
def setupEnvironmentForGPUs(args, setting):
    GPU = args.gpu
    N_GPU = len(GPU.split(","))

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = setting["init"]["TF_CPP_MIN_LOG_LEVEL"]

    K.set_floatx('float32')
    config = tf.compat.v1.ConfigProto()
    if setting["init"]["allow_growth"]:
        config.gpu_options.allow_growth = True
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices: tf.config.experimental.set_memory_growth(physical_device, True)

    config.gpu_options.per_process_gpu_memory_fraction = setting["init"]["per_process_gpu_memory_fraction"] * N_GPU
    tf.compat.v1.disable_eager_execution()
    # session = tf.compat.v1.Session(config=config)

    if is_verbose():
        print_sep("-", 50)
        print("Use {0} GPU(s): {1}".format(N_GPU, GPU))

    return N_GPU


################################################################################
# return the selected window for an image
def get_slice_window(img, startX, startY, constants, is_gt=False, remove_colorbar=False):
    M, N = constants["M"], constants["N"]
    sliceWindow = img[startX:startX+M,startY:startY+N]

    # check if there are any NaN elements
    if np.isnan(sliceWindow).any():
        where = list(map(list, np.argwhere(np.isnan(sliceWindow))))
        for w in where: sliceWindow[w] = constants["PIXELVALUES"][0]

    if is_gt:
        for pxval in constants["PIXELVALUES"]:
            sliceWindow = np.where(np.logical_and(
                sliceWindow>=np.rint(pxval-(256/6)), sliceWindow<=np.rint(pxval+(256/6))
            ), pxval, sliceWindow)
    # Remove the colorbar! starting coordinate: (129,435)
    if remove_colorbar and not is_ISLES2018():
        if M==constants["IMAGE_WIDTH"] and N==constants["IMAGE_HEIGHT"]: sliceWindow[:,colorbar_coord[1]:] = 0
        # if the tile is smaller than the entire image
        elif startY+N>=colorbar_coord[1]: sliceWindow[:,colorbar_coord[1]-startY:] = 0

    sliceWindow = np.cast["float32"](sliceWindow)  # cast the window into a float

    return sliceWindow


################################################################################
# Perform a data augmentation based on the index and return the image
def perform_DA_on_img(img, data_aug_idx):
    if data_aug_idx == 1: img = np.rot90(img)  # rotate 90 degree counterclockwise
    elif data_aug_idx == 2: img = np.rot90(img, 2)  # rotate 180 degree counterclockwise
    elif data_aug_idx == 3: img = np.rot90(img, 3)  # rotate 270 degree counterclockwise
    elif data_aug_idx == 4: img = np.flipud(img)  # rotate 270 degree counterclockwise
    elif data_aug_idx == 5: img = np.fliplr(img)  # flip the matrix left/right

    return img


################################################################################
# Get the epoch number from the partial weight filename
def getEpochFromPartialWeightFilename(partialWeightsPath):
    return int(partialWeightsPath[partialWeightsPath.index(suffix_partial_weights)+len(suffix_partial_weights):partialWeightsPath.index(".h5")])


################################################################################
# Get the loss defined in the settings
def get_loss(modelInfo):
    name = modelInfo["loss"]
    hyperparameters = modelInfo[name] if name in modelInfo.keys() else {}
    if name=="focal_tversky_loss": set_Focal_Tversky(hyperparameters)

    general_losses = {
        "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
        "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "mean_squared_error": tf.keras.losses.MeanSquaredError()
    }
    loss = {}

    if name in general_losses.keys(): loss["loss"] = general_losses[name]
    else: loss["loss"] = getattr(losses, name)
    loss["name"] = name

    if is_verbose(): print("[INFO] - Use {} Loss".format(name))
    return loss


################################################################################
# Get the statistic functions (& metrics) defined in the settings
def getMetricFunctions(listStats):
    general_metrics = [
        "binary_crossentropy",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_error",
        "accuracy"
    ]

    statisticFuncs = []
    for m in listStats: statisticFuncs.append(m) if m in general_metrics else statisticFuncs.append(getattr(metrics, m))

    if is_verbose(): print("[INFO] - Getting {} functions".format(listStats))
    if len(statisticFuncs)==0: statisticFuncs = None

    return statisticFuncs


################################################################################
# Return a flag to check if the filename (partial) is inside the list of patients
def isFilenameInListOfPatient(filename, patients, suffix):
    start_idx = filename.rfind(os.path.sep) + len(DATASET_PREFIX) + 1
    end_idx = filename.find(suffix)
    patient_id = filename[start_idx:end_idx]
    # don't load the dataframe if patient_id NOT in the list of patients
    if patient_id.find("_")==-1: patient_id = int(patient_id)
    ret = True if patient_id in patients else False
    return ret


################################################################################
# Get the correct class weights for the metrics
def getClassWeights(classtype):
    four_cat = [[1,1,0,0]] if classtype == "rest" else [[0, 0, 1, 0]] if classtype == "penumbra" else [[0, 0, 0, 1]]
    three_cat = [[1,0,0]] if classtype == "rest" else [[0, 1, 0]] if classtype == "penumbra" else [[0, 0, 1]]
    two_cat = [[1,0]] if classtype == "rest" else [[0, 1]]

    class_weights = tf.constant(four_cat, dtype=K.floatx())
    if get_n_classes() == 3: class_weights = tf.constant(three_cat, dtype=K.floatx())
    elif get_n_classes() == 2: class_weights = tf.constant(two_cat, dtype=K.floatx())
    return class_weights


################################################################################
################################################################################
################################################################################
########################### GENERAL UTILS ######################################
################################################################################
################################################################################
################################################################################
################################################################################


################################################################################
# get the string of the patient id given the integer
def get_str_from_idx(index):
    p_id = str(index)
    if len(p_id)==1: p_id = "0"+p_id
    return p_id


################################################################################
# return the suffix for the model and the patient dataset
def get_suffix():
    return "_" + str(get_slice_pixels()) + "_" + str(get_m()) + "x" + str(get_n()) + is_3D() + get_onetimepoint()


################################################################################
# get the full directory path, given a relative path
def get_dir_path(path):
    return get_rootpath() + path


################################################################################
# Generate a directory in dir_path
def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        if is_verbose(): print("[INFO] - Creating folder: " + dir_path)
        os.makedirs(dir_path)


################################################################################
# print a separation for verbose purpose
def print_sep(what, howmuch):
    print(what*howmuch)


################################################################################
# Convert the experiment number to a string of 3 letters
def convert_expnum_to_str(expnum):
    exp = str(expnum)
    while len(exp.split(".")[0])<3: exp = "0"+exp
    return exp


################################################################################
# Print the shape of the layer if we are in debug mode
def print_int_shape(layer):
    if is_verbose(): print(K.int_shape(layer))


################################################################################
def addPIDToWatchdog():
    # Add PID to watchdog list
    if ENABLE_WATCHDOG:
        if os.path.isfile(PID_WATCHDOG_PICKLE_PATH):
            PID_list_for_watchdog = pickle_load(PID_WATCHDOG_PICKLE_PATH)
            PID_list_for_watchdog.append(dict(pid=os.getpid()))
        else:
            PID_list_for_watchdog = [dict(pid=os.getpid())]

        # Save list
        pickle_save(PID_list_for_watchdog, PID_WATCHDOG_PICKLE_PATH)

        # Create a empty list for saving to when the model finishes
        if not os.path.isfile(PID_WATCHDOG_FINISHED_PICKLE_PATH):
            PID_list_finished_for_watchdog = []
            pickle_save(PID_list_finished_for_watchdog, PID_WATCHDOG_FINISHED_PICKLE_PATH)
    else:
        print('Warning: WATCHDOG IS DISABLED!')


################################################################################
def stopPIDToWatchdog():
    if ENABLE_WATCHDOG:
        # Add PID to finished-watchdog-list
        if os.path.isfile(PID_WATCHDOG_FINISHED_PICKLE_PATH):
            PID_list_finished_for_watchdog = pickle_load(PID_WATCHDOG_FINISHED_PICKLE_PATH)
            PID_list_finished_for_watchdog.append(dict(pid=os.getpid()))
            pickle_save(PID_list_finished_for_watchdog, PID_WATCHDOG_FINISHED_PICKLE_PATH)

        # Remove PID from watchdog list
        if os.path.isfile(PID_WATCHDOG_PICKLE_PATH):
            PID_list_for_watchdog = pickle_load(PID_WATCHDOG_PICKLE_PATH)
            PID_list_for_watchdog.remove(dict(pid=os.getpid()))
            pickle_save(PID_list_for_watchdog, PID_WATCHDOG_PICKLE_PATH)


################################################################################
def pickle_save(variable_to_save, path):
    with open(path, 'wb') as handle: pickle.dump(variable_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


################################################################################
def pickle_load(path):
    with open(path, 'rb') as handle: output = pickle.load(handle)
    return output


################################################################################
# Check memory usage of the model
def get_model_memory_usage(model, batch_size, first=True):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model': internal_model_mem_count += get_model_memory_usage(l, batch_size, first=False)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list: out_shape = out_shape[0]
        for s in out_shape:
            if s is None: continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    if is_verbose() and first:
        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

    number_size = 4.0
    if K.floatx() == 'float16': number_size = 2.0
    if K.floatx() == 'float64': number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

