#!/usr/bin/python

import sys

from Utils import general_utils, dataset_utils
import constants
from NeuralNetworkClass import NeuralNetwork


# MAIN FUNCTION
def main():
    networks = dict()
    train_df = None

    # Get the command line arguments
    args = general_utils.getCommandLineArguments()

    # Read the settings from json file
    setting = general_utils.getSettingFile(args.sname)

    # set up the environment for GPUs
    n_gpu = general_utils.setupEnvironment(args, setting)

    # initialize model(s)
    for name, info in setting["models"].items():
        networks[name] = NeuralNetwork(info, setting)

    for key, net in networks.items():
        for testPatient in setting["PATIENT_TO_TEST"]:
            p_id = general_utils.getStringPatientIndex(testPatient)
            # Training or Test ?

            # Check if the model was already trained and saved
            if net.isModelSaved(p_id):
                model = net.loadSavedModel(p_id)
            else:
                ## GET THE DATASET
                # if we are using a data augmentation dataset we need to get the dataset differently each time
                if net.da: train_df = general_utils.getDataset(net, p_id, multiprocessing=setting["init"]["multiprocessing"])
                else: # Otherwise get dataset only the first time
                    if train_df is None: train_df = general_utils.getDataset(net, multiprocessing=setting["init"]["multiprocessing"])

                ## PREPARE DATASET
                net.prepareDataset(train_df, p_id)
                ## SET THE CALLBACKS, RUN TRAINING & SAVE THE MODELS WEIGHTS
                net.setCallbacks(p_id)
                net.runTraining(p_id, n_gpu)
                net.saveModelAndWeight(p_id)

            ## PERFOM TESTING
            if net.supervised:
                net.evaluateModelWithCategorics(p_id)
            # predict and save the images
            net.predictAndSaveImages(p_id)






if __name__ == '__main__':
    main()
