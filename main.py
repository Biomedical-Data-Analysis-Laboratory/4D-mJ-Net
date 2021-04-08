#!/usr/bin/python

import os
import pickle
from Utils import general_utils, dataset_utils
from Model import training, constants
from Model.NeuralNetworkClass import NeuralNetwork
# to remove *SOME OF* the warning from tensorflow (regarding deprecation) <-- to remove when update tensorflow!
from tensorflow.python.util import deprecation
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

deprecation._PRINT_DEPRECATION_WARNINGS = False


################################################################################
# MAIN FUNCTION
def main():
    networks = list()
    train_df = None

    # Add the current PID to the watchdog list
    general_utils.addPIDToWatchdog()

    # Get the command line arguments
    args = general_utils.getCommandLineArguments()

    # Read the settings from json file
    setting = general_utils.getSettingFile(args.sname)
    if args.exp: setting["EXPERIMENT"] = args.exp

    # set up the environment for GPUs
    n_gpu = general_utils.setupEnvironment(args, setting)

    # initialize model(s)
    for info in setting["models"]: networks.append(NeuralNetwork(info, setting))

    for nn in networks:
        listOfPatientsToTrainVal = setting["PATIENTS_TO_TRAINVAL"]
        listOfPatientsToTest = list() if "PATIENTS_TO_TEST" not in setting.keys() else setting["PATIENTS_TO_TEST"]
        listOfPatientsToExclude = list() if "PATIENTS_TO_EXCLUDE" not in setting.keys() else setting["PATIENTS_TO_EXCLUDE"]

        # flag that states: run the train on all the patients in the "patient" folder
        if "ALL" == listOfPatientsToTrainVal[0]:
            # check if we want to get the dataset JUST based on the severity
            severity = listOfPatientsToTrainVal[0].split("_")[1] + "_" if "_" in listOfPatientsToTrainVal[0] else ""
            # different for SUS2020_v2 dataset since the dataset is not complete and the prefix is different
            if "SUS2020" in nn.datasetFolder:
                listOfPatientsToTrainVal = [d[len(constants.getPrefixImages()):] for d in
                                            os.listdir(nn.patientsFolder) if
                                            os.path.isdir(os.path.join(nn.patientsFolder, d)) and
                                            severity in d]
            else:
                listOfPatientsToTrainVal = [int(d[len(constants.getPrefixImages()):]) for d in
                                            os.listdir(nn.patientsFolder) if
                                            os.path.isdir(os.path.join(nn.patientsFolder, d)) and
                                            severity in d]

        # if DEBUG mode: use only 5 patients in the list
        if constants.getDEBUG():
            listOfPatientsToTrainVal = listOfPatientsToTrainVal[:5]
            listOfPatientsToTest = listOfPatientsToTest[:3]
            nn.setDebugDataset()

        listOfPatientsToTrainVal.sort()  # sort the list

        # remove the patients to exclude (if any)
        listOfPatientsToTest = list(set(listOfPatientsToTest) - set(listOfPatientsToExclude))
        listOfPatientsToTrainVal = list(set(listOfPatientsToTrainVal) - set(listOfPatientsToExclude))

        # loop over all the list of patients.
        # Useful for creating a model for each patient (if cross-validation is set)
        # else, it will create a unique model
        starting_rep, n_rep = 1, 1
        if nn.cross_validation["use"]:
            n_rep = nn.cross_validation["split"]
            starting_rep = nn.cross_validation["starting"] if "starting" in nn.cross_validation.keys() else 1

        for split_id in range(starting_rep,n_rep+1):
            nn.resetVars()
            model_split = general_utils.getStringFromIndex(split_id)
            nn.setModelSplit(model_split)

            # set the multi/single PROCESSING
            nn.setProcessingEnv(setting["init"]["MULTIPROCESSING"])

            # # GET THE DATASET:
            # - The dataset is composed of all the .pkl files in the dataset folder! (To load only once)
            if train_df is None: train_df = dataset_utils.getDataset(nn, listOfPatientsToTrainVal)
            val_list = nn.splitDataset(train_df, listOfPatientsToTrainVal, listOfPatientsToTest)

            # Check if the model was already trained and saved
            if nn.isModelSaved():
                # SET THE CALLBACKS & LOAD MODEL
                nn.setCallbacks()
                nn.loadSavedModel()
            else: nn.initializeAndStartTraining(n_gpu, args.jump)

            # TRAIN SET: only for ISLES2018 dataset
            if constants.getIsISLES2018(): nn.predictAndSaveImages([general_utils.getStringFromIndex(x) for x in listOfPatientsToTrainVal if x <1000], nn.isModelSaved())
            else: nn.predictAndSaveImages(val_list, nn.isModelSaved())  # VALIDATION SET: predict the images for decision on the model
            # PERFORM TESTING: predict and save the images
            nn.predictAndSaveImages(listOfPatientsToTest, nn.isModelSaved())

    general_utils.stopPIDToWatchdog()


################################################################################
################################################################################
if __name__ == '__main__':
    """
    Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-pm] [-t TILE] [-dim DIMENSION] [-c {2,3,4}] [-w ...] [-e EXP] [-j]

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use
      sname                 Select the setting filename

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -o, --original        Set the shape of the testing dataset to be compatible with the original shape 
                            (T,M,N) [time in front]
      -pm, --pm             Set the flag to train the parametric maps as input 
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM) (default = 16)
      -dim DIMENSION, --dimension DIMENSION
                            Set the dimension of the input images (width X height) (default = 512)
      -c {2,3,4}, --classes {2,3,4}
                            Set the # of classes involved (default = 4)
      -w, --weights         Set the weights for the categorical losses
      -e, --exp             Set the number of the experiment
      -j, --jump            Jump the training and go directly on the gradual fine-tuning function
      --timelast           Set the time dimension in the last channel of the input model          
    """
    main()
