import constants
from Utils import general_utils

import os, glob, random, time, random
import multiprocessing
import pandas as pd
import numpy as np
import hickle as hkl # Price et al., (2018). Hickle: A HDF5-based python pickle replacement. Journal of Open Source Software, 3(32), 1115, https://doi.org/10.21105/joss.01115


################################################################################
# Function to test the model `NOT USED OTHERWISE`
def initTestingDataFrame():
    patientList = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    testingList = []
    for sample in range(0, constants.SAMPLES*2):
        if sample<constants.SAMPLES:
            rand_pixels = np.random.randint(low=0, high=50, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            label = constants.LABELS[0]
            ground_truth = np.zeros(shape=(constants.getM(), constants.getN()))
        else:
            rand_pixels = np.random.randint(low=180, high=255, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.getM(), constants.getN()))
            label = constants.LABELS[1]
            ground_truth = np.ones(shape=(constants.getM(), constants.getN()))*255

        testingList.append((random.choice(patientList), label, rand_pixels, ground_truth))

    np.random.shuffle(testingList)
    # columns : ['patient_id', 'label', 'pixels', 'ground_truth']
    train_df = pd.DataFrame(testingList, columns=constants.dataFrameColumnsTest)
    train_df['label_code'] = train_df.label.map({constants.LABELS[0]:0, constants.LABELS[1]:1})

    return train_df

################################################################################
# Function to load the saved dataframe
def loadTrainingDataframe(net, testing_id=None):
    cpu_count = multiprocessing.cpu_count()
    # columns : ['patient_id', 'label', 'pixels', 'ground_truth', "label_code"]
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)

    suffix = general_utils.getSuffix()

    frames = [train_df]
    if not net.mp: # (SINGLE PROCESSING VERSION)
        for filename_train in glob.glob(net.datasetFolder+"*"+suffix+".hkl"):
            tmp_df = loadSingleTrainingData(net.da, filename_train, testing_id)
            frames.append(tmp_df)
    else: # MMULTI PROCESSING VERSION)
        input = []
        for filename_train in glob.glob(net.datasetFolder+"*"+suffix+".hkl"):
            input.append((net.da, filename_train, testing_id))

        with multiprocessing.Pool(processes=cpu_count) as pool: # auto closing workers
            frames = pool.starmap(loadSingleTrainingData, input)

    train_df = pd.concat(frames)

    # filename_train = SCRIPT_PATH+"train.h5"
    # if os.path.exists(filename_train):\
    #     print('Loading TRAIN dataframe from {0}...'.format(filename_train))
    #     train_df = pd.read_hdf(filename_train, "X_"+DTYPE+"_"+str(SAMPLES)+"_"+str(M)+"x"+str(N))

    return train_df

################################################################################
# Function to load a single dataframe from a patient index
def loadSingleTrainingData(da, filename_train, testing_id):
    #filename_train = SCRIPT_PATH+"trainComplete"+p_id+".h5"
    index = filename_train[-5:-3]
    p_id = general_utils.getStringPatientIndex(index)

    if constants.getVerbose(): print('Loading TRAIN dataframe from {}...'.format(filename_train))
    suffix = "_DATA_AUGMENTATION" if da else ""
    if testing_id==p_id:
         # take the normal dataset for the testing patient instead the augmented one...
        if constants.getVerbose(): print("---> Load normal dataset for patient {0}".format(testing_id))
        suffix= ""

    #tmp_df = readFromHDF(filename_train, suffix)
    tmp_df = readFromHickle(filename_train, suffix) # Faster and less space consuming!

    return tmp_df

################################################################################
# Return the elements in the filename saved as a hickle
def readFromHickle(filename, suffix):
    return hkl.load(filename)

################################################################################
# read the elements from filename and specific key
def readFromHDF(filename_train, suffix):
    return pd.read_hdf(filename_train, key="X_"+str(constants.getM())+"x"+str(constants.getN())+"_"+str(constants.SLICING_PIXELS) + suffix)

################################################################################
# Return the dataset based on the patient id
def getDataset(net, p_id=None):
    start = time.time()
    train_df = pd.DataFrame(columns=['patient_id', 'label', 'pixels', 'ground_truth', "label_code"])

    if constants.getVerbose():
        general_utils.printSeparation("-",50)
        if net.mp: print("Loading Dataset using MULTIprocessing...")
        else: print("Loading Dataset using SINGLEprocessing...")

    if constants.getDEBUG(): train_df = initTestingDataFrame()
    else:
        # no debugging and no data augmentation
        if net.da:
            print("Data augmented training/testing... load the dataset differently for each patient")
            train_df = loadTrainingDataframe(net, testing_id=p_id)
        else: train_df = loadTrainingDataframe(net)

    end = time.time()
    print("Total time to load the Dataset: {0}s".format(round(end-start, 3)))
    generateDatasetSummary(train_df)
    return train_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(nn, p_id):
    start = time.time()

    nn.dataset["train"]["indices"] = list()
    nn.dataset["val"]["indices"] = list()
    # train indices are ALL except the one = p_id
    train_val_dataset = np.nonzero((nn.train_df.patient_id.values != p_id))[0]

    if nn.val["random_validation_selection"]:
        # perform a random selection of the validation
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

    nn.dataset["train"]["data"] = getDataFromIndex(nn.train_df, nn.dataset["train"]["indices"], nn.mp)
    nn.dataset["val"]["data"] = None if nn.val["validation_perc"]==0 else getDataFromIndex(nn.train_df, nn.dataset["val"]["indices"], nn.mp)

    if nn.supervised:
        nn.dataset = getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)

    end = time.time()
    print("Total time to prepare the Dataset: {}s".format(round(end-start, 3)))

    return nn.dataset

################################################################################
# Get the test dataset
def getTestDataset(dataset, train_df, p_id, mp):
    # test indices are = p_id
    dataset["test"]["indices"] = np.nonzero((train_df.patient_id.values == p_id))[0]
    dataset["test"]["data"] = getDataFromIndex(train_df, dataset["test"]["indices"], mp)
    return dataset

################################################################################
# get the data from a list of indices
def getDataFromIndex(train_df, indices, mp):
    start = time.time()

    if not mp: # (SINGLE PROCESSING VERSION)
        data = np.array([np.array(a).reshape(constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN()) for a in train_df.pixels.values[indices]])
    else: # (MULTI PROCESSING VERSION)
        cpu_count = multiprocessing.cpu_count()
        input = [a for a in train_df.pixels.values[indices]]
        with multiprocessing.Pool(processes=cpu_count) as pool: # auto closing workers
            data = pool.map(getSingleDataFromIndex, input)
            data = np.array(data)

    end = time.time()
    print("*getDataFromIndex* Time: {}s".format(round(end-start, 3)))

    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1))
    return data

################################################################################

def getSingleDataFromIndex(singledata):
    return np.array(singledata).reshape(constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN())

################################################################################
# Return the labels ginve the indices
def getLabelsFromIndex(train_df, indices):
    labels = np.array([np.array(a).reshape(constants.getM(),constants.getN()) for a in train_df.ground_truth.values[indices]])
    # if SIGMOID_ACT: ??
    # convert the label in [0, 1] values
    labels = labels.astype("float32")
    labels /= 255

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
