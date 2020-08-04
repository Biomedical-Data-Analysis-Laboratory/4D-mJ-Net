# DO NOT import dataset_utils here!
import constants, training
from Utils import metrics, losses

import sys, argparse, os, json, time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

################################################################################
######################## UTILS FUNCTIONS #######################################
# The file should only contains functions!
################################################################################

################################################################################
# get the arguments from the command line
def getCommandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-d", "--debug", help="DEBUG mode", action="store_true")
    parser.add_argument("-o", "--original", help="ORIGINAL_SHAPE flag", action="store_true")
    parser.add_argument("-s", "--sname", help="Pass the setting filename")
    parser.add_argument("-t", "--tile", help="Set the tile pixels dimension (MxM)", type=int)
    parser.add_argument("-dim", "--dimension", help="Set the dimension of the input images (widthXheight)", type=int)
    parser.add_argument("-c", "--classes", help="Set the # of classe involved (default = 4)", default=4, type=int, choices=[2,3,4])
    parser.add_argument("gpu", help="Give the id of gpu (or a list of the gpus) to use")
    args = parser.parse_args()

    constants.setVerbose(args.verbose)
    constants.setDEBUG(args.debug)
    constants.setOriginalShape(args.original)
    constants.setTileDimension(args.tile)
    constants.setImageDimension(args.dimension)
    constants.setNumberOfClasses(args.classes)

    if not args.sname:
        args.sname = constants.default_setting_filename

    return args

################################################################################
# get the setting file
def getSettingFile(filename):
    setting = dict()

    # the path of the setting file start from the main.py
    # (= current working directory)
    with open(os.path.join(os.getcwd(), filename)) as f:
        setting = json.load(f)


    if constants.getVerbose():
        printSeparation("-",50)
        print("Load setting file: {}".format(filename))

    return setting

################################################################################
# setup the global environment
def setupEnvironment(args, setting):
    # important: set up the root path for later uses
    constants.setRootPath(setting["root_path"])

    if "NUMBER_OF_IMAGE_PER_SECTION" in setting["init"].keys(): constants.setImagePerSection(setting["init"]["NUMBER_OF_IMAGE_PER_SECTION"])
    if "3D" in setting["init"].keys() and setting["init"]["3D"]: constants.set3DFlag()
    if "ONE_TIME_POINT" in setting["init"].keys() and setting["init"]["ONE_TIME_POINT"]: constants.setONETIMEPOINT(getStringFromIndex(setting["init"]["ONE_TIME_POINT"]))

    experimentFolder = "EXP"+convertExperimentNumberToString(setting["EXPERIMENT"])+"/"
    N_GPU = setupEnvironmentForGPUs(args, setting)

    for key, rel_path in setting["relative_paths"].items():
        if isinstance(rel_path, dict):
            prefix = key.upper()+"/"
            createDir(prefix)
            createDir(prefix+experimentFolder)
            for sub_path in setting["relative_paths"][key].values():
                createDir(prefix+experimentFolder+sub_path)
        else:
            if rel_path!="": createDir(rel_path)

    return N_GPU

################################################################################
# setup the environment for the GPUs
def setupEnvironmentForGPUs(args, setting):
    GPU = args.gpu
    N_GPU = len(GPU.split(","))

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = setting["init"]["TF_CPP_MIN_LOG_LEVEL"]

    config = tf.compat.v1.ConfigProto()
    if setting["init"]["allow_growth"]:
        config.gpu_options.allow_growth = True
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices: tf.config.experimental.set_memory_growth(physical_device, True)

    config.gpu_options.per_process_gpu_memory_fraction = setting["init"]["per_process_gpu_memory_fraction"] * N_GPU
    session = tf.compat.v1.Session(config=config)

    if constants.getVerbose():
        printSeparation("-",50)
        print("Use {0} GPU(s): {1}".format(N_GPU, GPU))

    return N_GPU

################################################################################
# return the selected window for an image
def getSlicingWindow(img, startX, startY, M, N):
    return img[startX:startX+M,startY:startY+N]

################################################################################
# Get the epoch number from the partial weight filename
def getEpochFromPartialWeightFilename(partialWeightsPath):
    return int(partialWeightsPath[partialWeightsPath.index(constants.suffix_partial_weights)+len(constants.suffix_partial_weights):partialWeightsPath.index(".h5")])

################################################################################
# Get the loss defined in the settings
def getLoss(name):
    general_losses = ["binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mean_squared_error"]
    loss = {}

    if name in general_losses: loss["loss"] = name
    else: loss["loss"] = getattr(losses, name)
    loss["name"] = name

    if constants.getVerbose():
        printSeparation("-",50)
        print("[WARNING] - Use {} Loss".format(name))

    return loss

################################################################################
# Get the statistic functions (& metrics) defined in the settings
def getStatisticFunctions(listStats):
    statisticFuncs = []
    for m in listStats: statisticFuncs.append(getattr(metrics, m))

    if constants.getVerbose():
        printSeparation("-",50)
        print("[WARNING] - Getting {} functions".format(listStats))

    return statisticFuncs

################################################################################
# Return a flag to check if the filename (partial) is inside the list of patients
def isFilenameInListOfPatient(filename, patients):
    ret = False
    start_index = filename.rfind("/")+len(constants.DATASET_PREFIX)+1
    patient_id = filename[start_index:start_index+len(str(patients[-1]))]
    # don't load the dataframe if patient_id NOT in the list of patients
    if constants.PREFIX_IMAGES=="PA": patient_id = int(patient_id)
    if patient_id in patients: ret = True

    return ret

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
def getStringFromIndex(index):
    p_id = str(index)
    if len(p_id)==1: p_id = "0"+p_id
    return p_id

################################################################################
# return the suffix for the model and the patient dataset
def getSuffix():
    return "_"+str(constants.SLICING_PIXELS)+"_"+str(constants.getM())+"x"+str(constants.getN())+constants.get3DFlag()+constants.getONETIMEPOINT()

################################################################################
# get the full directory path, given a relative path
def getFullDirectoryPath(path):
    return constants.getRootPath()+path

################################################################################
# Generate a directory in dir_path
def createDir(dir_path):
    if not os.path.isdir(dir_path):
        if constants.getVerbose(): print("[INFO] - Creating folder: " + dir_path)
        os.makedirs(dir_path)

################################################################################
# print a separation for verbose purpose
def printSeparation(what, howmuch):
    print(what*howmuch)

################################################################################
# Convert the experiment number to a string of 3 letters
def convertExperimentNumberToString(expnum):
    exp = str(expnum)
    while len(exp)<3: exp = "0"+exp

    return exp

################################################################################
# Print the shaoe of the layer if we are in debug mode
def print_int_shape(layer):
    if constants.getDEBUG(): print(K.int_shape(layer))
