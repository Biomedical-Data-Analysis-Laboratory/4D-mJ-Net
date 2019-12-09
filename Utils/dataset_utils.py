import constants
from Utils import general_utils

import glob, random, time
import multiprocessing
import pandas as pd
import numpy as np

################################################################################
# Function to test the model `NOT USED OTHERWISE`
def initTestingDataFrame():
    patientList = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    testingList = []
    for sample in range(0, constants.SAMPLES*2):
        if sample<constants.SAMPLES:
            rand_pixels = np.random.randint(low=0, high=50, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.M, constants.N))
            label = constants.LABELS[0]
            ground_truth = np.zeros(shape=(constants.M, constants.N))
        else:
            rand_pixels = np.random.randint(low=180, high=255, size=(constants.NUMBER_OF_IMAGE_PER_SECTION, constants.M, constants.N))
            label = constants.LABELS[1]
            ground_truth = np.ones(shape=(constants.M, constants.N))*255

        testingList.append((random.choice(patientList), label, rand_pixels, ground_truth))

    np.random.shuffle(testingList)
    # columns : ['patient_id', 'label', 'pixels', 'ground_truth']
    train_df = pd.DataFrame(testingList, columns=constants.dataFrameColumnsTest)
    train_df['label_code'] = train_df.label.map({constants.LABELS[0]:0, constants.LABELS[1]:1})

    return train_df

################################################################################
# Function to load the saved dataframe
def loadTrainingDataframe(net, testing_id=None, multiprocessing=0):
    if multiprocessing==0:
        return loadTrainingDataframeSingleProcessing(net, testing_id)
    elif multiprocessing==1:
        return loadTrainingDataframeMultiProcessing(net, testing_id=None)

################################################################################
# Function to load the saved dataframe (SINGLE PROCESSING VERSION)
def loadTrainingDataframeSingleProcessing(net, testing_id=None):
    cpu_count = multiprocessing.cpu_count()
    # columns : ['patient_id', 'label', 'pixels', 'ground_truth', "label_code"]
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)

    frames = [train_df]

    for filename_train in glob.glob(net.datasetFolder+"*.h5"):
        tmp_df = loadSingleTrainingData(net.da, filename_train, testing_id)
        frames.append(tmp_df)

    train_df = pd.concat(frames)

    # filename_train = SCRIPT_PATH+"train.h5"
    # if os.path.exists(filename_train):
    #     print('Loading TRAIN dataframe from {0}...'.format(filename_train))
    #     train_df = pd.read_hdf(filename_train, "X_"+DTYPE+"_"+str(SAMPLES)+"_"+str(M)+"x"+str(N))

    return train_df

################################################################################
# Function to load the saved dataframe (MMULTI PROCESSING VERSION)
def loadTrainingDataframeMultiProcessing(net, testing_id=None):
    cpu_count = multiprocessing.cpu_count()
    # columns : ['patient_id', 'label', 'pixels', 'ground_truth', "label_code"]
    train_df = pd.DataFrame(columns=constants.dataFrameColumns)
    frames = [train_df]

    input = []
    for filename_train in glob.glob(net.datasetFolder+"*.h5"):
        input.append((net.da, filename_train, testing_id))

    with multiprocessing.Pool(processes=cpu_count) as pool: # auto closing workers
        results = pool.starmap(loadSingleTrainingData, input)

    train_df = pd.concat(results)

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

    tmp_df = pd.read_hdf(filename_train, key="X_"+str(constants.M)+"x"+str(constants.N)+"_"+str(constants.SLICING_PIXELS) + suffix)

    return tmp_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(dataset, train_df, validation_perc, supervised, p_id):
    start = time.time()
    val_mod = int(100/validation_perc)

    # TODO: this take a lot of time... more than loading the dataset...
    # try to use a multiprocessing! <--

    # train indices are ALL except the one = p_id
    train_val_dataset = np.nonzero((train_df.patient_id.values != p_id))[0]
    dataset["train"]["indices"] = np.nonzero((train_val_dataset%val_mod != 0))[0]
    dataset["val"]["indices"] = np.nonzero((train_val_dataset%val_mod == 0))[0]
    dataset["train"]["data"] = getDataFromIndex(train_df, dataset["train"]["indices"])
    dataset["val"]["data"] = getDataFromIndex(train_df, dataset["val"]["indices"])

    if supervised:
        # test indices are = p_id
        dataset["test"]["indices"] = np.nonzero((train_df.patient_id.values == p_id))[0]
        dataset["test"]["data"] = getDataFromIndex(train_df, dataset["test"]["indices"])

    end = time.time()
    print("Total time to prepare the Dataset: {0}s".format(round(end-start, 3)))

    return dataset

################################################################################
# get the data from a list of indices
def getDataFromIndex(train_df, indices):
    data = np.array([np.array(a).reshape(constants.M,constants.N,constants.NUMBER_OF_IMAGE_PER_SECTION) for a in train_df.pixels.values[indices]])
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1))
    return data

def getLabelsFromIndex(train_df, indices):
    labels = np.array([np.array(a).reshape(constants.M,constants.N) for a in train_df.ground_truth.values[indices]])
    # if SIGMOID_ACT: ??
    # convert the label in [0, 1] values
    labels = labels.astype("float32")
    labels /= 255

    return labels
