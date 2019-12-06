#!/usr/bin/python

import sys
import utils
import constants
from NeuralNetworkClass import NeuralNetwork


# MAIN FUNCTION
def main():
    networks = dict()

    # Get the command line arguments
    args = utils.getCommandLineArguments()

    # Read the settings from json file
    setting = utils.getSettingFile(args.sname)

    # set up the environment for GPUs
    utils.setupEnvironment(args, setting)

    # initialize model(s)
    for name, info in setting["models"].items():
        networks[name] = NeuralNetwork(info, setting)

    for key, net in networks.items():
        for testPatient in setting["PATIENT_TO_TEST"]:
            p_id = utils.getStringPatientIndex(testPatient)
            # Training or Test ?

            # Check if the model was already trained and saved
            if net.isModelSaved(p_id):
                model = net.loadSAvedModel(p_id)
            else:
                # get dataset
                print("not yet")
                # Run training

            # Perfom testing
            if net.supervised:
                print("not yet 2...")
                net.evaluateModelWithCategorics() # ...

            net.predictAndSaveImages(p_id)






if __name__ == '__main__':
    main()
