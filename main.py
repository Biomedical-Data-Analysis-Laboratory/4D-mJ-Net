#!/usr/bin/python

import os
from Utils import general_utils, dataset_utils
import constants
from NeuralNetworkClass import NeuralNetwork

# to remove *SOME OF* the warning from tensorflow (regarding deprecation) <-- to remove when update tensorflow!
from tensorflow.python.util import deprecation

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
    for info in setting["models"]: networks.append(NeuralNetwork(info, setting))

    for nn in networks:
        isAlreadySaved = False
        listOfPatientsToTrainVal = setting["PATIENTS_TO_TRAINVAL"]
        listOfPatientsToTest = list() if "PATIENTS_TO_TEST" not in setting.keys() else setting["PATIENTS_TO_TEST"]
        # flag that states: run the test on all the patients in the "patient" folder

        manual_annotationsFolder = os.path.join(constants.getRootPath(), nn.labeledImagesFolder)

        if "ALL" in listOfPatientsToTrainVal[0]:
            # check if we want to get the dataset JUST based on the severity
            severity = listOfPatientsToTrainVal[0].split("_")[1] + "_" if "_" in listOfPatientsToTrainVal[0] else ""

            # different for SUS2020_v2 dataset since the dataset is not complete and the prefix is different
            if "SUS2020" in nn.datasetFolder:
                listOfPatientsToTrainVal = [d[len(constants.getPrefixImages()):] for d in
                                            os.listdir(manual_annotationsFolder) if
                                            os.path.isdir(os.path.join(manual_annotationsFolder, d)) and
                                            severity in d]
            else:
                listOfPatientsToTrainVal = [int(d[len(constants.getPrefixImages()):]) for d in
                                            os.listdir(manual_annotationsFolder) if
                                            os.path.isdir(os.path.join(manual_annotationsFolder, d)) and
                                            severity in d]

        # if DEBUG mode: use only 5 patients in the list
        if constants.getDEBUG(): listOfPatientsToTrainVal = listOfPatientsToTrainVal[:5]
        listOfPatientsToTrainVal.sort()  # sort the list

        # loop over all the list of patients.
        # Useful for creating a model for each patient (if cross-validation is set)
        # else, it will create a unique model
        for testPatient in listOfPatientsToTrainVal:
            p_id = general_utils.getStringFromIndex(testPatient)

            # set the multi/single PROCESSING
            nn.setProcessingEnv(setting["init"]["MULTIPROCESSING"])

            # Check if the model was already trained and saved
            if nn.isModelSaved(p_id):
                # SET THE CALLBACKS & LOAD MODEL
                nn.setCallbacks(p_id)
                nn.loadSavedModel(p_id)
                isAlreadySaved = True
                break
            else:
                # # GET THE DATASET:
                # - The dataset is composed of all the .pkl files in the dataset folder! (To load only once)
                if train_df is None: train_df = dataset_utils.getDataset(nn, listOfPatientsToTrainVal)
                nn.splitDataset(train_df, p_id, listOfPatientsToTrainVal, listOfPatientsToTest)

                if nn.use_sequence:
                    # if we are doing a sequence train (for memory issue)
                    nn.prepareSequenceClass()
                    nn.runTrainSequence(p_id, n_gpu)
                else:
                    # # PREPARE DATASET (=divide in train/val/test)
                    nn.prepareDataset(p_id)
                    # # SET THE CALLBACKS, RUN TRAINING & SAVE THE MODELS WEIGHTS
                    if nn.train_on_batch: nn.runTrainingOnBatch(p_id, n_gpu)
                    else: nn.runTraining(p_id, n_gpu)

            nn.saveModelAndWeight(p_id)

        # # PERFORM TESTING: predict and save the images
        stats = nn.predictAndSaveImages(listOfPatientsToTest, isAlreadySaved)
        if nn.save_statistics: nn.saveStats(stats, "PATIENTS_TO_TEST")

        stats = nn.predictAndSaveImages(listOfPatientsToTrainVal, isAlreadySaved)
        if nn.save_statistics: nn.saveStats(stats, "PATIENTS_TO_TRAINVAL")


################################################################################
################################################################################
if __name__ == '__main__':
    """
    Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-s SETTING_FILENAME] [-t TILE] [-dim DIMENSION] [-c {2,3,4}]

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
    """
    main()
