import constants
from Utils import general_utils

import os, glob, random, time
import multiprocessing
import pandas as pd
import numpy as np
import pickle as pkl
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
    listOfFolders = glob.glob(nn.datasetFolder+"*"+suffix+".pkl")

    idx = 1
    for filename_train in listOfFolders:
        # don't load the dataframe if patient_id NOT in the list of patients
        if not general_utils.isFilenameInListOfPatient(filename_train, patients): continue

        if constants.getVerbose(): print('[INFO] - {0}/{1} Loading dataframe from {2}...'.format(idx, len(patients), filename_train))
        tmp_df = readFromPickle(filename_train)
        frames.append(tmp_df)

        idx+=1

    train_df = pd.concat(frames, sort=False)
    return train_df

################################################################################
# Return the elements in the filename saved as a pickle
def readFromPickle(filename):
    file = open(filename, "rb")
    return pkl.load(file)

################################################################################
# Return the dataset based on the patient id
# First function that is been called to create the train_df
def getDataset(nn, patients):
    start = time.time()

    if constants.getVerbose():
        general_utils.printSeparation("-",50)
        if nn.mp: print("[INFO] - Loading Dataset using MULTIprocessing...")
        else: print("[INFO] - Loading Dataset using SINGLEprocessing...")

    # Load the debug dataframe if we have set a debug flag or the real ones,
    # based on the list of patients
    if constants.getDEBUG(): train_df = initTestingDataFrame()
    else: train_df = loadTrainingDataframe(nn, patients=patients)

    end = time.time()
    print("[INFO] - Total time to load the Dataset: {0}s".format(round(end-start, 3)))
    generateDatasetSummary(train_df) # summary of the label in the dataset

    return train_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(nn, p_id, listOfPatientsToTest):
    start = time.time()

    for flag in ["train", "val", "test"]:
        nn.dataset[flag]["indices"] = list()

    if not nn.cross_validation:
        # here only if: NO cross-validation set
        validation_list, test_list = list(), list()

        # We have set a number of validation patient(s)
        if nn.val["number_patients_for_validation"]>0:
            validation_list = listOfPatientsToTest[:nn.val["number_patients_for_validation"]]
            if nn.getVerbose(): print("[INFO] - VALIDATION list: {}".format(validation_list))

            for val_p in validation_list:
                val_pid = general_utils.getStringFromIndex(val_p)
                nn.dataset["val"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == val_pid))[0])

            # We have set a number of testing patient(s) and we are inside a supervised learning
            if nn.val["number_patients_for_testing"]>0 and nn.supervised:
                test_list = listOfPatientsToTest[nn.val["number_patients_for_validation"]:nn.val["number_patients_for_validation"]+nn.val["number_patients_for_testing"]]
                if nn.getVerbose(): print("[INFO] - TEST list: {}".format(test_list))

                for test_p in test_list:
                    test_pid = general_utils.getStringFromIndex(test_p)
                    nn.dataset["test"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == test_pid))[0])
                # DEFINE the data for the dataset TEST
                nn.dataset["test"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["test"]["indices"], "test", nn.mp)

        # set the indices for the train dataset as the difference between all_indices, the validation indices and the test indices
        all_indices = np.nonzero((nn.train_df.label_code.values >= 0))[0]
        nn.dataset["train"]["indices"] = list(set(all_indices)-set(nn.dataset["val"]["indices"])-set(nn.dataset["test"]["indices"]))

    else: # We are doing a cross-validation!
        # train/val indices are ALL except the one = p_id
        if nn.getVerbose(): print("[INFO] - VALIDATION patient: {}".format(p_id))
        train_val_dataset = np.nonzero((nn.train_df.patient_id.values != p_id))[0]

        if nn.val["random_validation_selection"]: # perform a random selection of the validation
            val_mod = int(100/nn["val"]["validation_perc"])
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

            if nn.supervised: nn.dataset = getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)

    # set the train data only if we have NOT set the train_on_batch flag
    if not nn.train_on_batch: nn.dataset["train"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["train"]["indices"], "train", nn.mp)
    nn.dataset["val"]["data"] = None if nn.val["validation_perc"]==0 or nn.val["number_patients_for_validation"]==0 else getDataFromIndex(nn.train_df, nn.dataset["val"]["indices"], "val", nn.mp)

    end = time.time()
    if constants.getVerbose(): print("[INFO] - Total time to prepare the Dataset: {}s".format(round(end-start, 3)))

    return nn.dataset

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
    if type(data) is not np.ndarray: data = np.array(data)

    end = time.time()
    if constants.getVerbose():
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
        labels = data.astype("float32")
        labels /= 255 # convert the label in [0, 1] values

    end = time.time()
    if constants.getVerbose():
        print("[INFO] - *getLabelsFromIndex* Time: {}s".format(round(end-start, 3)))
        print("[INFO] - {0} shape; # {1}".format(labels.shape, flag))

    return labels

################################################################################
# Generate a summary of the dataset
def generateDatasetSummary(train_df):
    N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT = getNumberOfElements(train_df)

    general_utils.printSeparation('+', 90)
    print("DATASET SUMMARY: \n")
    print("\t N. Background: {0}".format(N_BACKGROUND))
    print("\t N. Brain: {0}".format(N_BRAIN))
    print("\t N. Penumbra: {0}".format(N_PENUMBRA))
    print("\t N. Core: {0}".format(N_CORE))
    print("\t Tot: {0}".format(N_TOT))
    general_utils.printSeparation('+', 90)

################################################################################
# Return the number of element per class of the dataset
def getNumberOfElements(train_df):
    N_BACKGROUND = len([x for x in train_df.label if x=="background"])
    N_BRAIN = len([x for x in train_df.label if x=="brain"])
    N_PENUMBRA = len([x for x in train_df.label if x=="penumbra"])
    N_CORE = len([x for x in train_df.label if x=="core"])
    N_TOT = train_df.shape[0]

    return (N_BACKGROUND, N_BRAIN, N_PENUMBRA, N_CORE, N_TOT)
