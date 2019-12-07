#!/usr/bin/python

import sys
import utils, constants, dataset_utils
from NeuralNetworkClass import NeuralNetwork


# MAIN FUNCTION
def main():
    networks = dict()

    # Get the command line arguments
    args = utils.getCommandLineArguments()

    # Read the settings from json file
    setting = utils.getSettingFile(args.sname)

    # set up the environment for GPUs
    n_gpu = utils.setupEnvironment(args, setting)

    # initialize model(s)
    for name, info in setting["models"].items():
        networks[name] = NeuralNetwork(info, setting)

    train_df = pd.DataFrame(columns=['patient_id', 'label', 'pixels', 'ground_truth', "label_code"])
    for key, net in networks.items():
        for testPatient in setting["PATIENT_TO_TEST"]:
            p_id = utils.getStringPatientIndex(testPatient)
            # Training or Test ?

            # Check if the model was already trained and saved
            if net.isModelSaved(p_id):
                model = net.loadSavedModel(p_id)
            else:
                ## GET THE DATASET
                # if we are using a data augmentation dataset we need to get the dataset differently each time
                if net.da: train_df = utils.getDataset(train_df, net, p_id)
                else: # Otherwise get dataset only the first time
                    if len(train_df.index) == 0: train_df = utils.getDataset(train_df, net)

                ## PREPARE DATASET
                net.prepareDataset(train_df, p_id)
                ## RUN TRAINING
                net.runTraining(p_id, n_gpu)


            ## PERFOM TESTING
            if net.supervised:
                net.evaluateModelWithCategorics()
            # predict and save the images
            net.predictAndSaveImages(p_id)






if __name__ == '__main__':
    main()
