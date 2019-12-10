# DO NOT import dataset_utils here!
import constants, training

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
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                    action="store_true")
    parser.add_argument("-s", "--sname", help="Pass the setting filename")
    parser.add_argument("gpu", help="Give the id of gpu (or a list of the gpus) to use")
    args = parser.parse_args()

    constants.setVerbose(args.verbose)

    if not args.sname:
        args.sname = constants.default_setting_filename

    return args

################################################################################
# get the setting file
def getSettingFile(filename):
    setting = dict()

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
    return int(partialWeightsPath[partialWeightsPath.index(":")+1:partialWeightsPath.index(".h5")])

################################################################################
# Get the loss defined in the settings
def getLoss(name):
    loss = {}

    if name=="dice_coef_loss":
        loss["loss"] = training.dice_coef_loss
        loss["name"] = "dice_coef"
    # elif .. # TODO:

    return loss

################################################################################
# Get the loss defined in the settings
def getMetrics(listMetrics):
    metrics = []

    for m in listMetrics:
        if m=="dice_coef":
            metrics.append(training.dice_coef)
        elif m=="sensitivity":
            metrics.append(training.sensitivity)
        elif m=="specificity":
            metrics.append(training.specificity)
        elif m=="auroc":
            metrics.append(training.aucroc)

    return metrics

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
