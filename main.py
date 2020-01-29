#!/usr/bin/python

import sys

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
        for testPatient in setting["PATIENT_TO_TEST"]:
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
                    if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = []

                    # meanV = np.mean(tmpStats[func.__name__][classToEval])
                    # stats[func.__name__][classToEval].append(meanV)
                    stats[func.__name__][classToEval].extend(tmpStats[func.__name__][classToEval])
        nn.saveStats(stats, "PATIENT_TO_TEST")

################################################################################
################################################################################
if __name__ == '__main__':
    main()
