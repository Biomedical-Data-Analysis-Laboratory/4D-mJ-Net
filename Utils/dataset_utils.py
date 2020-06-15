import constants
from Utils import general_utils

import os, glob, random, time
import multiprocessing
import pandas as pd
import numpy as np
import hickle as hkl # Price et al., (2018). Hickle: A HDF5-based python pickle replacement. Journal of Open Source Software, 3(32), 1115, https://doi.org/10.21105/joss.01115
from tensorflow.keras import utils
import tensorflow as tf

################################################################################
# Function to test the model `NOT USED OTHERWISE`
def initTestingDataFrame():
    patientList = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    testingList = []
    for sample in range(0, 1000):
        if sample<500:
            rand_pixels = np.random.randint(low=0, high=50, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            # rand_pixels = np.random.randint(low=0, high=50, size=(constants.getM(), constants.getN(), random.randrange(1,70)))
            label = constants.LABELS[0]
            ground_truth = np.zeros(shape=(constants.getM(), constants.getN()))
        else:
            rand_pixels = np.random.randint(low=180, high=255, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            # rand_pixels = np.random.randint(low=180, high=255, size=(constants.getM(), constants.getN(), random.randrange(1,70)))
            label = constants.LABELS[1]
            ground_truth = np.ones(shape=(constants.getM(), constants.getN()))*255

        testingList.append((random.choice(patientList), label, rand_pixels, ground_truth, (0,0)), sort=True)

    np.random.shuffle(testingList)
    train_df = pd.DataFrame(testingList, columns=constants.dataFrameColumns[:len(constants.dataFrameColumns)-1])
    train_df['label_code'] = train_df.label.map({constants.LABELS[0]:0, constants.LABELS[1]:1})

    return train_df

################################################################################
# Function to load the saved dataframe
def loadTrainingDataframe(nn, testing_id=None):
    cpu_count = multiprocessing.cpu_count()
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)
    # get the suffix based on the SLICING_PIXELS, the M and N
    suffix = general_utils.getSuffix() # es == "_4_16x16"

    frames = [train_df]
    if not nn.mp: # (SINGLE PROCESSING VERSION)
        for filename_train in glob.glob(nn.datasetFolder+"*"+suffix+".hkl"):
            tmp_df = loadSingleTrainingData(nn.da, filename_train, testing_id)
            frames.append(tmp_df)
    else: # (MULTI PROCESSING VERSION)
        input = []
        for filename_train in glob.glob(nn.datasetFolder+"*"+suffix+".hkl"):
            input.append((nn.da, filename_train, testing_id))

        with multiprocessing.Pool(processes=10) as pool: # auto closing workers
            frames = pool.starmap(loadSingleTrainingData, input)

    train_df = pd.concat(frames, sort=True)

    return train_df

################################################################################
# Function to load a single dataframe from a patient index
def loadSingleTrainingData(da, filename_train, testing_id):
    index = filename_train[-5:-3]
    p_id = general_utils.getStringPatientIndex(index)

    if constants.getVerbose(): print('[INFO] - Loading dataframe from {}...'.format(filename_train))
    suffix = "_DATA_AUGMENTATION" if da else ""
    if testing_id==p_id:
         # take the normal dataset for the testing patient instead the augmented one...
        if constants.getVerbose(): print("---> Load normal dataset for patient {0}".format(testing_id))
        suffix= ""

    #tmp_df = readFromHDF(filename_train, suffix)
    tmp_df = readFromHickle(filename_train) # Faster and less space consuming!

    return tmp_df

################################################################################
# Return the elements in the filename saved as a hickle
def readFromHickle(filename):
    return hkl.load(filename)

################################################################################
# read the elements from filename and specific key
def readFromHDF(filename_train, suffix):
    return pd.read_hdf(filename_train, key="X_"+str(constants.getM())+"x"+str(constants.getN())+"_"+str(constants.SLICING_PIXELS) + suffix)

################################################################################
# Return the dataset based on the patient id
# First function that is been called to create the train_df
def getDataset(nn, p_id=None):
    start = time.time()
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)

    if constants.getVerbose():
        general_utils.printSeparation("-",50)
        if nn.mp: print("[INFO] - Loading Dataset using MULTIprocessing...")
        else: print("[INFO] - Loading Dataset using SINGLEprocessing...")

    if constants.getDEBUG(): train_df = initTestingDataFrame()
    else:
        # no debugging and no data augmentation
        if nn.da:
            print("[WARNING] - Data augmented training/testing... load the dataset differently for each patient")
            train_df = loadTrainingDataframe(nn, testing_id=p_id)
        else: train_df = loadTrainingDataframe(nn)

    end = time.time()
    print("[INFO] - Total time to load the Dataset: {0}s".format(round(end-start, 3)))
    generateDatasetSummary(train_df)
    return train_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(nn, p_id, listOfPatientsToTest):
    start = time.time()

    nn.dataset["train"]["indices"] = list()
    nn.dataset["val"]["indices"] = list()

    if not nn.cross_validation and nn.val["number_patients_for_validation"]>0 and nn.val["number_patients_for_testing"]>0:
        # here only if: NO cross-validation and the flags for number of patients (test/val) is > 0
        validation_list = listOfPatientsToTest[:nn.val["number_patients_for_validation"]]
        val_indices = set()
        test_list = listOfPatientsToTest[nn.val["number_patients_for_validation"]:nn.val["number_patients_for_validation"]+nn.val["number_patients_for_testing"]]

        for val_p in validation_list:
            current_pid = general_utils.getStringPatientIndex(val_p)
            nn.dataset["val"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == current_pid))[0])

        for test_p in test_list:
            current_pid = general_utils.getStringPatientIndex(test_p)
            if nn.supervised:
                if "indices" not in nn.dataset["test"].keys(): nn.dataset["test"]["indices"] = list()
                nn.dataset["test"]["indices"].extend(np.nonzero((nn.train_df.patient_id.values == current_pid))[0])

        nn.dataset["test"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["test"]["indices"], "test", nn.mp)

        all_indices = np.nonzero((nn.train_df.label_code.values >= 0))[0]
        nn.dataset["train"]["indices"] = list(set(all_indices)-set(nn.dataset["val"]["indices"])-set(nn.dataset["test"]["indices"]))

    else:
        # train indices are ALL except the one = p_id
        train_val_dataset = np.nonzero((nn.train_df.patient_id.values != p_id))[0]

        if nn.val["random_validation_selection"]: # perform a random selection of the validation
            val_mod = int(100/nn["val"]["validation_perc"])
            nn.dataset["train"]["indices"] = np.nonzero((train_val_dataset%val_mod != 0))[0]
            nn.dataset["val"]["indices"] = np.nonzero((train_val_dataset%val_mod == 0))[0]
        else:
            # do NOT use a patient(s) as a validation set because maybe it doesn't have
            # too much information about core and penumbra.
            # Instead, get a percentage from each class!
            for classLabelName in constants.LABELS:
                random.seed(0)
                classIndices = np.nonzero((nn.train_df.label.values[train_val_dataset]==classLabelName))[0]
                classValIndices = [] if nn.val["validation_perc"]==0 else random.sample(list(classIndices), int((len(classIndices)*nn.val["validation_perc"])/100))
                nn.dataset["train"]["indices"].extend(list(set(classIndices)-set(classValIndices)))
                if nn.val["validation_perc"]!=0: nn.dataset["val"]["indices"].extend(classValIndices)

            if nn.supervised: nn.dataset = getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)

    nn.dataset["train"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["train"]["indices"], "train", nn.mp)
    nn.dataset["val"]["data"] = None if nn.val["validation_perc"]==0 else getDataFromIndex(nn.train_df, nn.dataset["val"]["indices"], "val", nn.mp)

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

    input = [a for a in np.array(train_df.pixels.values[indices], dtype=object)]
    with multiprocessing.Pool(processes=10) as pool: # auto closing workers
        data = pool.map(getSingleDataFromIndex, input)

    if type(data) is not np.array: data = np.array(data)

    end = time.time()
    if constants.getVerbose():
        print("[INFO] - *getDataFromIndex* Time: {}s".format(round(end-start, 3)))
        print("[INFO] - {0} shape; # {1}".format(data.shape, flag))

    return data

################################################################################

def getSingleDataFromIndex(singledata):
    return singledata.reshape(singledata.shape + (1,))

def getSingleLabelFromIndex(singledata):
    return singledata.reshape(constants.getM(),constants.getN())

def getSingleLabelFromIndexCateg(singledata):
    return np.array(utils.to_categorical(np.trunc((singledata/256)*len(constants.LABELS)), num_classes=len(constants.LABELS)))

################################################################################
# Return the labels given the indices
def getLabelsFromIndex(train_df, indices, to_categ, flag):
    start = time.time()
    labels = None

    data = [a for a in np.array(train_df.ground_truth.values[indices])]
    if to_categ:
        with multiprocessing.Pool(processes=10) as pool: # auto closing workers
            labels = pool.map(getSingleLabelFromIndexCateg, data)
        if type(labels) is not np.array: labels = np.array(labels)
    else:
        with multiprocessing.Pool(processes=10) as pool: # auto closing workers
            labels = pool.map(getSingleLabelFromIndex, data)
        if type(labels) is not np.array: labels = np.array(labels)
        labels = labels.astype("float32")
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
