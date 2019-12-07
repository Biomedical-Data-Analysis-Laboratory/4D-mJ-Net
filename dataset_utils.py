import constants

import random
import pandas as pd
import numpy as np

################################################################################
# Function to test the model `NOT USED OTHERWISE`
def initTestingDataFrame():
    patientList = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    testingList = []
    for sample in range(0, constants.SAMPLES*2):
        if sample<constants.SAMPLES:
            rand_pixels = np.random.randint(0, 50, (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.M, constants.N))
            label = constants.LABELS[0]
        else:
            rand_pixels = np.random.randint(180, 255, (constants.NUMBER_OF_IMAGE_PER_SECTION, constants.M, constants.N))
            label = constants.LABELS[1]

        testingList.append((sample, random.choice(patientList), label, rand_pixels, 100))

    np.random.shuffle(testingList)
    train_df = pd.DataFrame(testingList, columns=['ID', 'patient_id', 'label', 'pixels', 'percentage'])
    train_df['label_code'] = train_df.label.map({constants.LABELS[0]:0, constants.LABELS[1]:1})

    return train_df

################################################################################
# Function to lead the saved dataframe
def loadTrainingDataframe(net, testing_id=None):
    train_df = pd.DataFrame(columns=['patient_id', 'label', 'pixels', 'ground_truth', "label_code"])

    frames = [train_df]
    for filename_train in glob.glob(net.datasetFolder+"*/"):
        #filename_train = SCRIPT_PATH+"trainComplete"+p_id+".h5"
        index = filename_train[-5:-3]
        p_id = getStringPatientIndex(index)

        #if os.path.exists(filename_train):
        if constants.getVerbose(): print('Loading TRAIN dataframe from {}...'.format(filename_train))
        suffix = "_DATA_AUGMENTATION" if net.da else ""
        if testing_id==p_id:
             # take the normal dataset for the testing patient instead the augmented one...
            if constants.getVerbose(): print("---> Load normal dataset for patient {0}".format(testing_id))
            suffix= ""

        tmp_df = pd.read_hdf(filename_train, key="X_"+str(constants.M)+"x"+str(constants.N)+"_"+str(constants.SLICING_PIXELS) + suffix)
        frames.append(tmp_df)
    train_df = pd.concat(frames)

    # filename_train = SCRIPT_PATH+"train.h5"
    # if os.path.exists(filename_train):
    #     print('Loading TRAIN dataframe from {0}...'.format(filename_train))
    #     train_df = pd.read_hdf(filename_train, "X_"+DTYPE+"_"+str(SAMPLES)+"_"+str(M)+"x"+str(N))

    return train_df

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
def prepareDataset(dataset, train_df, validation_perc, supervised, p_id):
    val_mod = 100/validation_perc

    # train indices are ALL except the one = p_id
    dataset["train"]["indices"] = np.nonzero((train_df.patient_id.values != p_id))[0]
    dataset["train"]["indices"] = np.nonzero((dataset["train"]["indices"]%val_mod != 0))
    dataset["val"]["indices"] = np.nonzero((dataset["train"]["indices"]%val_mod == 0))[0]
    dataset["train"]["data"] = getDataFromIndex(train_df, dataset["train"]["indices"])
    dataset["val"]["data"] = getDataFromIndex(train_df, dataset["val"]["indices"])

    if supervised:
        # test indices are = p_id
        dataset["test"]["indices"] = np.nonzero((train_df.patient_id.values == p_id))[0]
        dataset["test"]["data"] = getDataFromIndex(train_df, dataset["test"]["indices"])

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
    retun labels /= 255
