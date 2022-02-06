#!/usr/bin/python

import os
from Utils import general_utils, dataset_utils
from Model.constants import *
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
    args = general_utils.get_commandline_args()

    # Read the settings from json file
    setting = general_utils.get_setting_file(args.sname)
    if args.exp: setting["EXPERIMENT"] = args.exp

    # set up the environment for GPUs
    n_gpu = general_utils.setup_env(args, setting)

    # initialize model(s)
    for info in setting["models"]: networks.append(NeuralNetwork(info, setting))

    for nn in networks:
        patientlist_train_val = setting["PATIENTS_TO_TRAINVAL"]
        patientlist_test = list() if "PATIENTS_TO_TEST" not in setting.keys() else setting["PATIENTS_TO_TEST"]
        patientlist_exclude = list() if "PATIENTS_TO_EXCLUDE" not in setting.keys() else setting["PATIENTS_TO_EXCLUDE"]

        # flag that states: run the train on all the patients in the "patient" folder
        if "ALL" == patientlist_train_val[0]:
            # check if we want to get the dataset JUST based on the severity: i.e. --> "ALL_01"
            severity = patientlist_train_val[0].split("_")[1] + "_" if "_" in patientlist_train_val[0] else ""
            # different for SUS2020_v2 dataset since the dataset is not complete and the prefix is different
            if "SUS2020" in nn.ds_folder:
                patientlist_train_val = [d[len(DATASET_PREFIX):-(len(general_utils.get_suffix())+4)]
                                         for d in os.listdir(nn.ds_folder) if severity in d]
            else: patientlist_train_val = [int(d[len(get_prefix_img()):]) for d in
                                           os.listdir(nn.patients_folder) if
                                           os.path.isdir(os.path.join(nn.patients_folder, d)) and
                                           severity in d]

        # if DEBUG mode: use only a fix number of patients in the list
        if is_debug():
            patientlist_train_val = patientlist_train_val[:20]
            patientlist_test = patientlist_test[:3]
            nn.set_debug_ds()

        patientlist_train_val.sort()  # sort the list

        # remove the patients to exclude (if any)
        patientlist_test = list(set(patientlist_test) - set(patientlist_exclude))
        patientlist_train_val = list(set(patientlist_train_val) - set(patientlist_exclude))

        # loop over all the list of patients.
        # Useful for creating a model for each patient (if cross-validation is set)
        # else, it will create a unique model
        starting_rep, n_rep = 1, 1
        if nn.cross_validation["use"]:
            n_rep = nn.cross_validation["split"]
            starting_rep = nn.cross_validation["starting"] if "starting" in nn.cross_validation.keys() else 1

        for split_id in range(starting_rep,n_rep+1):
            nn.reset_vars()
            model_split = general_utils.get_str_from_idx(split_id)
            nn.set_model_split(model_split)

            # set the multi/single PROCESSING
            nn.set_processing_env(setting["init"]["MULTIPROCESSING"])

            # # GET THE DATASET:
            # - The dataset is composed of all the .pkl files in the dataset folder! (To load only once)
            if train_df is None: train_df = dataset_utils.get_ds(nn, patientlist_train_val)
            val_list = nn.split_ds(train_df, patientlist_train_val, patientlist_test)

            # Check if the model was already trained and saved
            if nn.is_model_saved():
                # SET THE CALLBACKS & LOAD MODEL
                nn.set_callbacks()
                nn.load_saved_model()
            else: nn.init_and_start_training(n_gpu, args.jump)

            # TRAIN SET: only for ISLES2018 dataset
            if is_ISLES2018():
                nn.predict_and_save_img([general_utils.get_str_from_idx(x) for x in patientlist_train_val if x < 1000], nn.is_model_saved())
            else: nn.predict_and_save_img(val_list, nn.is_model_saved())  # VALIDATION SET: predict the images for decision on the model
            # PERFORM TESTING: predict and save the images
            nn.predict_and_save_img(patientlist_test, nn.is_model_saved())

    general_utils.stopPIDToWatchdog()


################################################################################
################################################################################
if __name__ == '__main__':
    """
    Usage: python main.py gpu sname
                [-h] [-v] [-d] [-o] [-pm] [-t TILE] [-dim DIMENSION] [-c {2,3,4}] [-w ...] [-e EXP] [-j]  [--timelast] [--prefix PREFIX]

    positional arguments:
      gpu                   Give the id of gpu (or a list of the gpus) to use
      sname                 Select the setting filename

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Increase output verbosity
      -d, --debug           DEBUG mode
      -o, --original        Set the shape of the testing dataset to be compatible with the original shape 
      -pm, --pm             Set the flag to train the parametric maps as input 
      -t TILE, --tile TILE  Set the tile pixels dimension (MxM) (default = 16)
      -dim DIMENSION, --dimension DIMENSION
                            Set the dimension of the input images (width X height) (default = 512)
      -c {2,3,4}, --classes {2,3,4}
                            Set the # of classes involved (default = 4)
      --isles2018           Flag to use the ISLES2018 dataset
      -w, --weights         Set the weights for the categorical losses
      -e, --exp             Set the number of the experiment
      -j, --jump            Jump the training and go directly on the gradual fine-tuning function
      --timelast            Set the time dimension in the last channel of the input model          
      --prefix              Set the prefix different from the default
      --limcols             Set the columns without additional info 
    """
    main()
