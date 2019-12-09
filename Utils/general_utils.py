from Utils import dataset_utils
import constants

import sys, argparse, os, json, time
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
        printSeparation("-",50)

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
        printSeparation("-",50)

    return N_GPU

################################################################################
# Return the dataset based
def getDataset(net, p_id=None, multiprocessing=0):
    start = time.time()
    train_df = pd.DataFrame(columns=['patient_id', 'label', 'pixels', 'ground_truth', "label_code"])

    if constants.getVerbose():
        printSeparation("-",50)
        if multiprocessing: print("Loading Dataset using MULTIprocessing...")
        else: print("Loading Dataset using SINGLEprocessing...")
        printSeparation("-",50)

    if constants.DEBUG: train_df = dataset_utils.initTestingDataFrame()
    else:
        # no debugging and no data augmentation
        if net.da:
            print("Data augmented training/testing... load the dataset differently for each patient")
            train_df = dataset_utils.loadTrainingDataframe(net, testing_id=p_id, multiprocessing=multiprocessing)
        else: train_df = dataset_utils.loadTrainingDataframe(net, multiprocessing=multiprocessing)

    end = time.time()
    print("Total time to load the Dataset: {0}s".format(round(end-start, 3)))
    generateDatasetSummary(train_df)
    return train_df

################################################################################
# return the selected window for an image
def getSlicingWindow(img, startX, startY, M, N):
    return img[startX:startX+M,startY:startY+N]

################################################################################
# Generate a summary of the dataset
def generateDatasetSummary(train_df):
    N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT = getNumberOfElements(train_df)

    printSeparation('+', 90)
    print("DATASET SUMMARY: \n")
    print("\t N. Background: {0}".format(N_BACKGROUND))
    print("\t N. Brain: {0}".format(N_BRAIN))
    print("\t N. Penumbra: {0}".format(N_PENUMBRA))
    print("\t N. Core: {0}".format(N_CORE))
    print("\t Tot: {0}".format(N_TOT))
    printSeparation('+', 90)

################################################################################
# Return the number of element per class of the dataset
def getNumberOfElements(train_df):
    N_BACKGROUND = len([x for x in train_df.label if x=="background"])
    N_BRAIN = len([x for x in train_df.label if x=="brain"])
    N_PENUMBRA = len([x for x in train_df.label if x=="penumbra"])
    N_CORE = len([x for x in train_df.label if x=="core"])
    N_TOT = train_df.shape[0]

    return (N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT)

################################################################################
# Get the epoch number from the partial weight filename
def getEpochFromPartialWeightFilename(partialWeightsPath):
    return int(partialWeightsPath[partialWeightsPath.index(":")+1:partialWeightsPath.index(".h5")])

def getLoss(name):
    loss = {}

    if name=="dice_coef":
        loss["loss"] = dice_coef_loss
        loss["metrics"] = dice_coef
        loss["name"] = name
    # elif .. # TODO:

    return loss

################################################################################
# Funtion that calculates the DICE coefficient. Important when calculates the different of two images
def dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + 1) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + 1)

################################################################################
# Function that calculates the DICE coefficient loss. Util for the LOSS function during the training of the model (for image in input and output)!
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

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
