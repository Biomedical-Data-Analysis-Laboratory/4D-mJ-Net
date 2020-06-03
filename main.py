#!/usr/bin/python

import sys
import os
from Utils import general_utils, dataset_utils
import constants
from NeuralNetworkClass import NeuralNetwork

# to remove *SOME OF* the warning from tensorflow (regarding deprecation) <-- to remove when update tensorflow!
from tensorflow.python.util import deprecation
import numpy as np
deprecation._PRINT_DEPRECATION_WARNINGS = False

################################################################################
# MAIN FUNCTION
def main():
    networks = list()
    train_df = None

    # Get the command line arguments
    args = general_utils.getCommandLineArguments()

    # Read the settings from json file
    setting = general_utils.getSettingFile(args.sname)

    # set up the environment for GPUs
    n_gpu = general_utils.setupEnvironment(args, setting)

    # initialize model(s)
    for info in setting["models"]:
        networks.append(NeuralNetwork(info, setting))

    for nn in networks:
        stats = {}

        listOfPatientsToTest = setting["PATIENT_TO_TEST"]
        if listOfPatientsToTest[0] == "ALL": # flag that states: runn the test on all the patients in the "patient" folder
            mainPatsFolder = os.path.join(constants.getRootPath(),nn.patientsFolder)
            listOfPatientsToTest = [int(d[len(constants.PREFIX_IMAGES):]) for d in os.listdir(mainPatsFolder) if os.path.isdir(os.path.join(mainPatsFolder, d))]

        for testPatient in listOfPatientsToTest:
            p_id = general_utils.getStringPatientIndex(testPatient)
            isAlreadySaved = False

            # set the multi/single PROCESSING
            nn.setProcessingEnv(setting["init"]["MULTIPROCESSING"])

            # Check if the model was already trained and saved
            if nn.isModelSaved(p_id):
                # SET THE CALLBACKS & LOAD MODEL
                nn.setCallbacks(p_id)
                nn.loadSavedModel(p_id)
                isAlreadySaved = True
            else:
                ## GET THE DATASET:
                # - The dataset is composed of all the .hkl (or .h5) files in the dataset folder!
                # if we are using a data augmentation dataset we need to get the dataset differently each time
                if nn.da: train_df = dataset_utils.getDataset(nn, p_id)
                else: # Otherwise get dataset only the first time
                    if train_df is None: train_df = dataset_utils.getDataset(nn)
                ## PREPARE DATASET
                nn.prepareDataset(train_df, p_id)
                ## SET THE CALLBACKS, RUN TRAINING & SAVE THE MODELS WEIGHTS
                nn.runTraining(p_id, n_gpu)
                nn.saveModelAndWeight(p_id)

            ## PERFOM TESTING
            if nn.supervised:
                nn.evaluateModelWithCategorics(p_id, isAlreadySaved)
            # predict and save the images
            tmpStats = nn.predictAndSaveImages(p_id)

            for func in nn.statistics:
                for classToEval in nn.classes_to_evaluate:
                    if func.__name__ not in stats.keys(): stats[func.__name__] = {}
                    if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = {}
                    for idxE, _ in enumerate(nn.epsiloList):
                        if idxE not in stats[func.__name__][classToEval].keys(): stats[func.__name__][classToEval][idxE] = []
                        stats[func.__name__][classToEval][idxE].extend(tmpStats[func.__name__][classToEval][idxE])
        nn.saveStats(stats, "PATIENT_TO_TEST")

################################################################################
################################################################################
if __name__ == '__main__':
    """
    Usage: python main.py [-h] [-v] [-d] [-s SNAME] [-t TILE] gpu

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -s SNAME, --sname SNAME
                            Pass the setting filename
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM)

    """
    main()
