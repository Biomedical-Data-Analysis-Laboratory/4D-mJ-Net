# DO NOT import dataset_utils here!
import constants, training
from Utils import metrics, losses

import sys, argparse, os, json, time
import numpy as np
import pandas as pd
import tensorflow as tf

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
    parser.add_argument("-s", "--sname", help="Pass the setting filename")
    parser.add_argument("-t", "--tile", help="Set the tile pixels dimension (MxM)")
    parser.add_argument("gpu", help="Give the id of gpu (or a list of the gpus) to use")
    args = parser.parse_args()

    constants.setVerbose(args.verbose)
    constants.setDEBUG(args.debug)
    constants.setTileDimension(args.tile)

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
    constants.setRootPath(setting["root_path"])
    N_GPU = setupEnvironmentForGPUs(args, setting)

    for key, rel_path in setting["relative_paths"].items():
        if isinstance(rel_path, dict):
            createDir(key.upper()+"/")
            for sub_path in setting["relative_paths"][key].values():
                createDir(key.upper()+"/"+sub_path)
        else:
            createDir(rel_path)

    return N_GPU

################################################################################
# setup the environment for the GPUs
def setupEnvironmentForGPUs(args, setting):
    GPU = args.gpu
    N_GPU = len(GPU.split(","))

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = setting["init"]["TF_CPP_MIN_LOG_LEVEL"]

    config = tf.compat.v1.ConfigProto() #tf.ConfigProto()
    if setting["init"]["allow_growth"]: config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = setting["init"]["per_process_gpu_memory_fraction"] * N_GPU
    session = tf.compat.v1.Session(config=config) #tf.Session(config=config)

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
    loss = {}

    loss["loss"] = getattr(losses, name)
    loss["name"] = name

    # if name=="dice_coef_loss":
    #     loss["loss"] = losses.dice_coef_loss
    #     loss["name"] = "dice_coef"
    # elif name=="jaccard_index_loss":
    #     loss["loss"] = losses.jaccard_index_loss
    #     loss["name"] = "jaccard_distance"
    # elif name=="mod_dice_coef_loss":
    #     loss["loss"] = losses.mod_dice_coef_loss
    #     loss["name"] = "mod_dice_coef"
    # elif name=="dice_coef_binary_loss":
    #     loss["loss"] = losses.dice_coef_binary_loss
    #     loss["name"] = "dice_coef_binary"
    # elif name=="generalized_dice_loss":
    #     loss["loss"] = losses.generalized_dice_loss
    #     loss["name"] = "generalized_dice_coeff"
    # elif name=="tversky_loss":
    #     loss["loss"] = losses.tversky_loss
    #     loss["name"] = "tversky_loss"

    return loss

################################################################################
# Get the statistic functions (& metrics) defined in the settings
def getStatisticFunctions(listStats):
    statisticFuncs = []

    for m in listStats:
        statisticFuncs.append(getattr(metrics, m))
        
        # if m=="dice_coef":
        #     statisticFuncs.append(getattr(metrics,"dice_coef"))
        # elif m=="jaccard_distance":
        #     statisticFuncs.append(getattr(metrics,"jaccard_distance"))
        # elif m=="sensitivity" or m=="recall":
        #     statisticFuncs.append(getattr(metrics,"sensitivity"))
        # elif m=="specificity":
        #     statisticFuncs.append(getattr(metrics,"specificity"))
        # elif m=="accuracy":
        #     statisticFuncs.append(getattr(metrics,"accuracy"))
        # elif m=="precision":
        #     statisticFuncs.append(getattr(metrics,"precision"))
        # elif m=="f1":
        #     statisticFuncs.append(getattr(metrics,"f1"))
        # elif m=="jaccard_index":
        #     statisticFuncs.append(getattr(metrics,"jaccard_index"))
        # elif m=="mod_dice_coef":
        #     statisticFuncs.append(getattr(metrics,"mod_dice_coef"))
        # elif m=="dice_coef_binary":
        #     statisticFuncs.append(getattr(metrics,"dice_coef_binary"))
        # elif m=="generalized_dice_coeff":
        #     statisticFuncs.append(getattr(metrics,"generalized_dice_coeff"))

    return statisticFuncs

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
def getStringPatientIndex(patient_index):
    p_id = str(patient_index)
    if len(p_id)==1: p_id = "0"+p_id

    return p_id

################################################################################
# return the suffix for the model and the patient dataset
def getSuffix():
    return "_"+str(constants.SLICING_PIXELS)+"_"+str(constants.getM())+"x"+str(constants.getN())

################################################################################
# get the full directory path, given a relative path
def getFullDirectoryPath(path):
    return constants.getRootPath()+path

################################################################################
# Generate a directory in dir_path
def createDir(dir_path):
    if not os.path.isdir(dir_path): os.makedirs(dir_path)

################################################################################
# print a separation
def printSeparation(what, howmuch):
    print(what*howmuch)
