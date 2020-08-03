import constants
from Utils import general_utils

import os, glob, random, time
import multiprocessing
import pandas as pd
import numpy as np
import pickle as pkl
import hickle as hkl
import sklearn
from tensorflow.keras import utils
import tensorflow as tf

################################################################################
# Function to test the model `NOT USED OTHERWISE`
def initTestingDataFrame():
    # patientList = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    patientList = ["00_001", "00_002", "01_008", "01_005", "01_016", "01_017", "02_008", "01_049", "02_010", "00_011"]
    testingList = []
    for sample in range(100):
        if sample<50:
            # rand_pixels = np.random.randint(low=0, high=50, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            rand_pixels = np.random.randint(low=0, high=50, size=(constants.getM(), constants.getN(), 3))
            label = constants.LABELS[0]
            ground_truth = np.zeros(shape=(constants.getM(), constants.getN()))
        else:
            # rand_pixels = np.random.randint(low=180, high=255, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            rand_pixels = np.random.randint(low=180, high=255, size=(constants.getM(), constants.getN(), 3))
            label = constants.LABELS[1]
            ground_truth = np.ones(shape=(constants.getM(), constants.getN()))*255

        testingList.append((random.choice(patientList), label, rand_pixels, ground_truth, (0,0), 0, 0, 0))

    np.random.shuffle(testingList)
    train_df = pd.DataFrame(testingList, columns=constants.dataFrameColumns[:len(constants.dataFrameColumns)-1])
    train_df['label_code'] = train_df.label.map({constants.LABELS[0]:0, constants.LABELS[1]:1})

    return train_df

################################################################################
# Function to load the saved dataframes based on the list of patients
def loadTrainingDataframe(nn, patients, testing_id=None):
    cpu_count = multiprocessing.cpu_count()
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)
    # get the suffix based on the SLICING_PIXELS, the M and N
    suffix = general_utils.getSuffix() # es == "_4_16x16"

    frames = [train_df]

    suffix_filename = ".pkl"
    if nn.use_hickle: suffix_filename = ".hkl"
    listOfFolders = glob.glob(nn.datasetFolder+"*"+suffix+suffix_filename)

    idx = 1
    for filename_train in listOfFolders:
        # don't load the dataframe if patient_id NOT in the list of patients
        if not general_utils.isFilenameInListOfPatient(filename_train, patients): continue

        if constants.getVerbose(): print('[INFO] - {0}/{1} Loading dataframe from {2}...'.format(idx, len(patients), filename_train))

        tmp_df = readFromPickleOrHickle(filename_train, nn.use_hickle)
        frames.append(tmp_df)

        idx+=1

    train_df = pd.concat(frames, sort=False)
    return train_df

################################################################################
# Return the elements in the filename saved as a pickle or as hickle (depending on the flag)
def readFromPickleOrHickle(filename, flagHickle):
    if flagHickle: return sklearn.utils.shuffle(hkl.load(filename))
    else:
        file = open(filename, "rb")
        return sklearn.utils.shuffle(pkl.load(file))

################################################################################
################################################################################
# Return the dataset based on the patient id
# First function that is been called to create the train_df!
def getDataset(nn, listOfPatientsToTrainVal):
    start = time.time()

    if constants.getVerbose():
        general_utils.printSeparation("-",50)
        if nn.mp: print("[INFO] - Loading Dataset using MULTIprocessing...")
        else: print("[INFO] - Loading Dataset using SINGLEprocessing...")

    # Load the debug dataframe if we have set a debug flag or the real ones,
    # based on the list of patients
    if constants.getDEBUG(): train_df = initTestingDataFrame()
    else: train_df = loadTrainingDataframe(nn, patients=listOfPatientsToTrainVal)

    end = time.time()
    print("[INFO] - Total time to load the Dataset: {0}s".format(round(end-start, 3)))
    if constants.getVerbose(): generateDatasetSummary(train_df, listOfPatientsToTrainVal) # summary of the dataset

    return train_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(nn, p_id, listOfPatientsToTrainVal, listOfPatientsToTest):
    start = time.time()
    validation_list, test_list = list(), list()

    for flag in ["train", "val", "test"]: nn.dataset[flag]["indices"] = list()

    if not nn.cross_validation: # here only if: NO cross-validation set
        # We have set a number of validation patient(s)
        if nn.val["number_patients_for_validation"]>0:
            random.seed(0) # use ALWAYS the same random indices
            validation_list = random.sample(listOfPatientsToTrainVal, nn.val["number_patients_for_validation"])
            # remove the validation_list elements from the list
            listOfPatientsToTrainVal = list(set(listOfPatientsToTrainVal).difference(validation_list))
            if nn.getVerbose(): print("[INFO] - VALIDATION list: {}".format(validation_list))

            for val_p in validation_list:
                val_pid = general_utils.getStringFromIndex(val_p)
                nn.dataset["val"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == val_pid))[0])

        # We have set a number of testing patient(s) and we are inside a supervised learning
        if nn.supervised:
            if nn.val["number_patients_for_testing"]>0 or len(listOfPatientsToTest)>0:
                test_list = list()
                if len(listOfPatientsToTest)>0: test_list = listOfPatientsToTest
                else:
                    random.seed(0) # use ALWAYS the same random indices
                    test_list = random.sample(listOfPatientsToTrainVal, nn.val["number_patients_for_testing"])
                # remove the validation_list elements from the list
                listOfPatientsToTrainVal = list(set(listOfPatientsToTrainVal).difference(test_list))
                if nn.getVerbose(): print("[INFO] - TEST list: {}".format(test_list))

                for test_p in test_list:
                    test_pid = general_utils.getStringFromIndex(test_p)
                    nn.dataset["test"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == test_pid))[0])
                # DEFINE the data for the dataset TEST
                nn.dataset["test"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["test"]["indices"], "test", nn.mp)

        # set the indices for the train dataset as the difference between all_indices, the validation indices and the test indices
        all_indices = np.nonzero((nn.train_df.label_code.values >= 0))[0]
        nn.dataset["train"]["indices"] = list(set(all_indices)-set(nn.dataset["val"]["indices"])-set(nn.dataset["test"]["indices"]))

        # the validation list is empty, the test list contains some patient(s) and the validation_perc > 0
        if len(validation_list)==0 and len(test_list)>0 and nn.val["validation_perc"]>0:
            train_val_dataset = nn.dataset["train"]["indices"]
            nn.dataset["train"]["indices"] = list() # empty the indices, it will be set inside the next function
            nn = getRandomOrWeightedValidationSelection(nn, train_val_dataset, test_list, p_id)

    else: # We are doing a cross-validation!
        # train/val indices are ALL except the one == p_id
        if nn.getVerbose(): print("[INFO] - VALIDATION patient: {}".format(p_id))
        train_val_dataset = np.nonzero((nn.train_df.patient_id.values != p_id))[0]
        nn = getRandomOrWeightedValidationSelection(nn, train_val_dataset, test_list, p_id)

    # set the train data only if we have NOT set the train_on_batch flag
    if not nn.train_on_batch: nn.dataset["train"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["train"]["indices"], "train", nn.mp)
    # the validation data is None if validation_perc and number_patients_for_validation are BOTH equal to 0
    nn.dataset["val"]["data"] = None if nn.val["validation_perc"]==0 and nn.val["number_patients_for_validation"]==0 else getDataFromIndex(nn.train_df, nn.dataset["val"]["indices"], "val", nn.mp)

    end = time.time()
    if constants.getVerbose(): print("[INFO] - Total time to prepare the Dataset: {}s".format(round(end-start, 3)))

    return nn.dataset

################################################################################
# Return the train and val indices based on a random selection (val_mod) if nn.val["random_validation_selection"]
# or a weighted selection based on the percentage (nn.val["validation_perc"]) for each class label
def getRandomOrWeightedValidationSelection(nn, train_val_dataset, test_list, p_id):
    # perform a random selection of the validation
    if nn.val["random_validation_selection"]:
        val_mod = int(100/nn.val["validation_perc"])
        nn.dataset["train"]["indices"] = np.nonzero((train_val_dataset%val_mod != 0))[0]
        nn.dataset["val"]["indices"] = np.nonzero((train_val_dataset%val_mod == 0))[0]
    else:
        # do NOT use a patient(s) as a validation set because maybe it doesn't have
        # too much information about core and/or penumbra. Instead, get a percentage from each class!
        for classLabelName in constants.LABELS:
            random.seed(0) # use ALWAYS the same random indices
            classIndices = np.nonzero((nn.train_df.label.values[train_val_dataset]==classLabelName))[0]
            classValIndices = [] if nn.val["validation_perc"]==0 else random.sample(list(classIndices), int((len(classIndices)*nn.val["validation_perc"])/100))
            nn.dataset["train"]["indices"].extend(list(set(classIndices)-set(classValIndices)))
            if nn.val["validation_perc"]>0: nn.dataset["val"]["indices"].extend(classValIndices)

        # if we are in a supervised environment and the the test_list is empty, update the test dataset
        if nn.supervised and len(test_list)==0: nn.dataset = getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)

    return nn

################################################################################
# Get the test dataset, where the test indices are == p_id
def getTestDataset(dataset, train_df, p_id, mp):
    dataset["test"]["indices"] = np.nonzero((train_df.patient_id.values == p_id))[0]
    dataset["test"]["data"] = getDataFromIndex(train_df, dataset["test"]["indices"], "test", mp)
    return dataset

################################################################################
# get the data from a list of indices
def getDataFromIndex(train_df, indices, flag, mp):
    start = time.time()

    if constants.get3DFlag()!="": data = [a for a in np.array(train_df.pixels.values[indices], dtype=object)]
    else:  # do this when NO 3D flag is set
        # reshape the data adding a last (1,)
        data = [a.reshape(a.shape + (1,)) for a in np.array(train_df.pixels.values[indices], dtype=object)]

    # convert the data into an np.ndarray
    if type(data) is not np.ndarray: data = np.array(data, dtype=np.uint8)

    end = time.time()
    if constants.getVerbose():
        setPatients = set(train_df.patient_id.values[indices])
        print("[INFO] - patients: {0}".format(setPatients))
        print("[INFO] - *getDataFromIndex* Time: {}s".format(round(end-start, 3)))
        print("[INFO] - {0} shape; # {1}".format(data.shape, flag))

    return data

################################################################################
# Fuction that reshape the data in a MxN tile
def getSingleLabelFromIndex(singledata):
    return singledata.reshape(constants.getM(),constants.getN())

################################################################################
# Fuction that convert the data into a categorical array based on the number of classes
def getSingleLabelFromIndexCateg(singledata):
    return np.array(utils.to_categorical(np.trunc((singledata/256)*len(constants.LABELS)), num_classes=len(constants.LABELS)))

################################################################################
# Return the labels given the indices
def getLabelsFromIndex(train_df, dataset, modelname, to_categ, flag):
    start = time.time()
    labels = None
    indices = dataset["indices"]

    # if we are using an autoencoder, the labels are the same as the data!
    if modelname.find("autoencoder")>-1: return dataset["data"]

    data = [a for a in np.array(train_df.ground_truth.values[indices])]

    if to_categ:
        with multiprocessing.Pool(processes=1) as pool: # auto closing workers
            labels = pool.map(getSingleLabelFromIndexCateg, data)
        if type(labels) is not np.array: labels = np.array(labels)
    else:
        if constants.N_CLASSES==3:
            for i,curr_data in enumerate(data):
                data[i][curr_data==255] = constants.PIXELVALUES[0] # remove one class from the ground truth
                data[i][curr_data==150] = constants.PIXELVALUES[2] # change the class for core
        if type(data) is not np.array: data = np.array(data)
        labels = data.astype(np.float32)
        labels /= 255 # convert the label in [0, 1] values

    end = time.time()
    if constants.getVerbose():
        print("[INFO] - *getLabelsFromIndex* Time: {}s".format(round(end-start, 3)))
        print("[INFO] - {0} shape; # {1}".format(labels.shape, flag))

    return labels

################################################################################
# Generate a summary of the dataset
def generateDatasetSummary(train_df, listOfPatientsToTrainVal):
    N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT = getNumberOfElements(train_df)

    general_utils.printSeparation('+', 100)
    print("DATASET SUMMARY: \n")
    print("\t N. Background: {0}".format(N_BACKGROUND))
    print("\t N. Brain: {0}".format(N_BRAIN))
    print("\t N. Penumbra: {0}".format(N_PENUMBRA))
    print("\t N. Core: {0}".format(N_CORE))
    print("\t Tot: {0}".format(N_TOT))
    print("\t Patients: {0}".format(listOfPatientsToTrainVal))
    general_utils.printSeparation('+', 100)

################################################################################
# Return the number of element per class of the dataset
def getNumberOfElements(train_df):
    N_BACKGROUND = len([x for x in train_df.label if x=="background"])
    N_BRAIN = len([x for x in train_df.label if x=="brain"])
    N_PENUMBRA = len([x for x in train_df.label if x=="penumbra"])
    N_CORE = len([x for x in train_df.label if x=="core"])
    N_TOT = train_df.shape[0]

    return (N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT)
