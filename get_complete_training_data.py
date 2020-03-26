#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import cv2
import time
import glob
import numpy as np
import pandas as pd
import os
import operator
import random
import hickle as hkl # Price et al., (2018). Hickle: A HDF5-based python pickle replacement. Journal of Open Source Software, 3(32), 1115, https://doi.org/10.21105/joss.01115



# #### CONSTANTS

# In[2]:


ROOT_PATH = "/home/stud/lucat/PhD_Project/Stroke_segmentation/"
#ROOT_PATH = "/Users/lucatomasetti/Desktop/uni-stavanger/MASTER_THESIS/"
SCRIPT_PATH = "/local/home/lucat/DATASET/" # "/local/home/lucat/DATASET/oldPatients/"


LABELLED_IMAGES_FOLDER_LOCATION = ROOT_PATH + "Manual_annotations/"
SAVE_REGISTERED_FOLDER = ROOT_PATH + "PATIENTS/" # "OLDPREPROC_PATIENTS/"

NUMBER_OF_IMAGE_PER_SECTION = 30 # number of image (divided by time) for each section of the brain
IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
# background:255, brain:0, penumbra:~76, core:~150
LABELS = ["background", "brain", "penumbra", "core"]
dataset, listPatientsDataset, trainDatasetList = {}, {}, list()


# In[3]:


DATA_AUGMENTATION = True


# In[4]:


M, N = int(IMAGE_WIDTH/16), int(IMAGE_HEIGHT/16)
SLICING_PIXELS = int(M/4)

PERCENTAGE_BACKGROUND_IMAGES = 20


# ### Util Classes

# Class for the slicing window

# In[5]:


class AreaInImage():
    def __init__(self, matrix, label):
        self.imgMatrix = matrix
        self.listOfStartingPoints = []
        self.slicingWindow = {}
        self.label = label

    def appendInListOfStartingPoints(self, points):
        self.listOfStartingPoints.append(points)


# ### Util functions

# In[6]:


def initializeLabels(patientIndex):
    global dataset
    dataset = dict() # reset the dataset
    dataset[patientIndex] = dict()

    dataset[patientIndex]["data"] = list()
    dataset[patientIndex]["label_class"] = list()
    dataset[patientIndex]["ground_truth"] = list()


# In[7]:


def getLabelledAreas(patientIndex, timeIndex):
    return cv2.imread(LABELLED_IMAGES_FOLDER_LOCATION+"Patient"+patientIndex+"/"+patientIndex+timeIndex+".png", 0)


# Function that return the slicing window from `img`, starting at pixels `startX` and `startY` with a width of `M` and height of `N`.

# In[8]:


def getSlicingWindow(img, startX, startY, M, N):
    return img[startX:startX+M,startY:startY+N]


# Function ofr inserting inside `dataset` the pixel areas (slicing windows) found with a slicing area approach (= start from point 0,0 it takes the areas `MxN` and then move on the right or on the bottom by a pre-fixed number of pixels = `SLICING_PIXELS`) and the corresponding area in the images inside the same folder, which are the registered images of the same section of the brain in different time.

# In[9]:


def fillDataset(train_df, relativePath, patientIndex, timeFolder):
    global dataset

    timeIndex = timeFolder.replace(SAVE_REGISTERED_FOLDER+relativePath, '').replace("/", "")
    if len(timeIndex)==1: timeIndex="0"+timeIndex

    labelledMatrix = getLabelledAreas(patientIndex, timeIndex)

    numBack, numBrain, numPenumbra, numCore = 0, 0, 0, 0
    startingX, startingY = 0, 0
    tmpListPixels, tmpListClasses, tmpListGroundTruth = list(), list(), list()
    backgroundPixelList, backgroundGroundTruthList = list(), list()

    imagesDict = {} # faster access to the images
    for imagename in np.sort(glob.glob(timeFolder+"*.png")): # sort the images !
        filename = imagename.replace(timeFolder, '')
        # don't take the first image (the manually annotated one)
        if "OLDPREPROC_PATIENTS/" in SAVE_REGISTERED_FOLDER and filename == "01.png": continue

        image = cv2.imread(imagename, 0)
        imagesDict[filename] = image

    pixelsList, otherInforList = list(), list()
    numRep = 1
    if DATA_AUGMENTATION: numRep = 6

    for rep in range(numRep):
        pixelsList.append(dict())
        otherInforList.append(dict())

    while True:
        if startingX>=IMAGE_WIDTH-M and startingY>=IMAGE_HEIGHT-N: # if we reach the end of the image, break the while loop.
            break

        realLabelledWindow = getSlicingWindow(labelledMatrix, startingX, startingY, M, N)
        binaryBackgroundMatrix = realLabelledWindow>=250
        binaryBrainMatrix = realLabelledWindow>=0
        binaryPenumbraMatrix = realLabelledWindow>=30
        binaryCoreMatrix = realLabelledWindow>=100

        valueClasses = dict()

        # extract the core area but not the brain area (= class 3)
        binaryCoreNoSkull = binaryBackgroundMatrix ^ binaryCoreMatrix # background XOR core
        valueClasses[LABELS[3]] = sum(sum(binaryCoreNoSkull))
        # extract the penumbra area but not the brain area (= class 2)
        binaryPenumbraNoSkull = binaryCoreMatrix ^ binaryPenumbraMatrix # penumbra XOR core
        valueClasses[LABELS[2]] = sum(sum(binaryPenumbraNoSkull))
        # extract the brain area but not the background (= class 1)
        binaryBrainMatrixNoBackground = binaryBrainMatrix ^ binaryPenumbraMatrix # brain XOR penumbra
        valueClasses[LABELS[1]] = sum(sum(binaryBrainMatrixNoBackground))
        # (= class 0)
        valueClasses[LABELS[0]] = sum(sum(binaryBackgroundMatrix))

        # the max of these values is the class to set for the binary class (Y)
        classToSet = max(valueClasses.items(), key=operator.itemgetter(1))[0]

        # set the window with just the four classes
        realLabelledWindow = (binaryBackgroundMatrix*255)+(binaryCoreNoSkull*150)+(binaryPenumbraNoSkull*76)+(binaryBrainMatrixNoBackground*0)

        numReplication = 1
        if classToSet==LABELS[0]: numBack+=1
        elif classToSet==LABELS[1]: numBrain+=1
        elif classToSet==LABELS[2]: numPenumbra+=1
        elif classToSet==LABELS[3]:
            numReplication = 6 if DATA_AUGMENTATION else 1
            numCore+=numReplication

        for data_aug_idx in range(numReplication):
            # tmparray = []

            for imagename in np.sort(glob.glob(timeFolder+"*.png")): # sort the images !
                if str(startingX) not in pixelsList[data_aug_idx].keys(): pixelsList[data_aug_idx][str(startingX)] = dict()
                if str(startingX) not in otherInforList[data_aug_idx].keys(): otherInforList[data_aug_idx][str(startingX)] = dict()
                if str(startingY) not in pixelsList[data_aug_idx][str(startingX)].keys(): pixelsList[data_aug_idx][str(startingX)][str(startingY)] = list()
                if str(startingY) not in otherInforList[data_aug_idx][str(startingX)].keys(): otherInforList[data_aug_idx][str(startingX)][str(startingY)] = dict()

                filename = imagename.replace(timeFolder, '')
                # don't take the first image (the manually annotated one)
                if "OLDPREPROC_PATIENTS/" in SAVE_REGISTERED_FOLDER and  filename == "01.png": continue

                image = imagesDict[filename]
                slicingWindow = getSlicingWindow(image, startingX, startingY, M, N)

                if data_aug_idx==1: slicingWindow = np.rot90(slicingWindow) # rotate 90 degree counterclockwise
                elif data_aug_idx==2: slicingWindow = np.rot90(slicingWindow,2) # rotate 180 degree counterclockwise
                elif data_aug_idx==3: slicingWindow = np.rot90(slicingWindow,3) # rotate 270 degree counterclockwise
                elif data_aug_idx==4: slicingWindow = np.flipud(slicingWindow) # flip the matrix up/down
                elif data_aug_idx==5: slicingWindow = np.fliplr(slicingWindow) # flip the matrix left/right

                # tmparray.extend(slicingWindow)

                pixelsList[data_aug_idx][str(startingX)][str(startingY)].extend(slicingWindow)

            otherInforList[data_aug_idx][str(startingX)][str(startingY)]["ground_truth"] = realLabelledWindow
            otherInforList[data_aug_idx][str(startingX)][str(startingY)]["label_class"] = classToSet

            # if classToSet==LABELS[0]:
            #     backgroundPixelList.append(tmparray)
            #     backgroundGroundTruthList.append(realLabelledWindow)
            # else:
            #     tmpListPixels.append(tmparray)
            #     tmpListGroundTruth.append(realLabelledWindow)
            #     tmpListClasses.append(classToSet)

        if startingY<IMAGE_HEIGHT-N: startingY += SLICING_PIXELS
        else:
            if startingX<IMAGE_WIDTH-M:
                startingY = 0
                startingX += SLICING_PIXELS

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("\t\t\t Background: {0}".format(numBack))
    print("\t\t\t Brain: {0}".format(numBrain))
    print("\t\t\t Penumbra: {0}".format(numPenumbra))
    print("\t\t\t Core: {0}".format(numCore))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print(train_df.shape)

    backElem = 0
    for d in range(0, len(pixelsList)):
        for x in pixelsList[d].keys():
            for y in pixelsList[d][x].keys():
                pixels = np.array(pixelsList[d][x][y]).reshape(NUMBER_OF_IMAGE_PER_SECTION,M,N)
                label = otherInforList[d][x][y]["label_class"]
                gt = otherInforList[d][x][y]["ground_truth"]

                perc = random.randint(0, 100)
                if label==LABELS[0]:
                    if perc > PERCENTAGE_BACKGROUND_IMAGES:
                        continue
                    else:
                        backElem+=1
                tmp_df = pd.DataFrame(np.array([[patientIndex, label, pixels, gt]]), columns=['patient_id', 'label', 'pixels', 'ground_truth'])
                tmp_df['label_code'] = tmp_df.label.map({LABELS[0]:0, LABELS[1]:1, LABELS[2]:2, LABELS[3]:3})
                train_df = train_df.append(tmp_df, ignore_index=True)

    print(train_df.shape)

    # indices = random.sample(range(0,len(backgroundPixelList)), MAX_NUM_BACKGROUND_IMAGES)
    # newBackgroundPixelList, newBackgroundGroundTruthList = list(), list()
    # for index in indices:
    #     newBackgroundPixelList.append(backgroundPixelList[index])
    #     newBackgroundGroundTruthList.append(backgroundGroundTruthList[index])

    print("\t\t\t Randomly picked {0} background images (~ {1} %).".format(str(backElem), PERCENTAGE_BACKGROUND_IMAGES))

    # dataset[patientIndex]["data"] = tmpListPixels + newBackgroundPixelList
    # dataset[patientIndex]["ground_truth"] = tmpListGroundTruth + newBackgroundGroundTruthList
    # dataset[patientIndex]["label_class"] = tmpListClasses + [LABELS[0]]*MAX_NUM_BACKGROUND_IMAGES

    return train_df

# Function that initialize the dataset: for each subfolder of the patient (section of the brain), it call the `fillDataset` function to get the pixels, save into the dataset and analyze them later.

# In[10]:


def initializeDataset():
    patientFolders = glob.glob(SAVE_REGISTERED_FOLDER+"*/")

    suffix_filename = "_"+str(SLICING_PIXELS)+"_"+str(M)+"x"+str(N)

    for numFold, patientFolder in enumerate(patientFolders): # for each patient
        train_df = pd.DataFrame(columns=['patient_id', 'label', 'pixels', 'ground_truth', 'label_code']) # reset the dataframe for every patient

        relativePath = patientFolder.replace(SAVE_REGISTERED_FOLDER, '')
        patientIndex = relativePath.replace("PA", "").replace("/", "")
        #filename_train = SCRIPT_PATH+"trainComplete"+str(patientIndex)+".h5"
        filename_train = SCRIPT_PATH+"patient"+str(patientIndex)+suffix_filename+".hkl"
        subfolders = glob.glob(patientFolder+"*/")

        ## only patients from 2 to 11 (the one with annotations)
        if int(patientIndex)>1 and int(patientIndex)<=11:
            print("\t Analyzing {0}/{1}; patient folder: {2}...".format(numFold+1, len(patientFolders), relativePath))
            for count, timeFolder in enumerate(subfolders): # for each slicing time
                initializeLabels(patientIndex)
                print("\t\t Analyzing subfolder {0}".format(timeFolder.replace(SAVE_REGISTERED_FOLDER, '').replace(relativePath, '')))
                start = time.time()

                train_df = fillDataset(train_df, relativePath, patientIndex, timeFolder) # insert the data inside the dataset dictionary
                # print("\t\t Details:", [(key, len(subdataset)) for key, subdataset in dataset[patientIndex].items()])
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("\t\t Background: {0}".format(str(sum(train_df.label==LABELS[0]))))
                print("\t\t Brain: {0}".format(str(sum(train_df.label==LABELS[1]))))
                print("\t\t Penumbra: {0}".format(str(sum(train_df.label==LABELS[2]))))
                print("\t\t Core: {0}".format(str(sum(train_df.label==LABELS[3]))))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

                end = time.time()
                print("\t\t Processed {0}/{1} subfolders in {2}s.".format(count+1, len(subfolders), round(end-start, 3)))

                # start = time.time()
                # convertDatasetInList()
                # print("Preparing partial TRAIN dataframe...")
                # tmp = prepareTraining()
                # end = time.time()
                # append the new rows in the dataframe
                # for index in range(0, tmp.shape[0]): train_df = train_df.append( tmp.iloc[index] )
                # print("Time: {0}".format(round(end-start, 3)))
                print("Train shape: ", train_df.shape)
            print("Saving TRAIN dataframe for patient {1} in {0}...".format(filename_train, str(patientIndex)))
            #suffix = "_DATA_AUGMENTATION" if DATA_AUGMENTATION else ""
            #train_df.to_hdf(filename_train, key="X_"+str(M)+"x"+str(N)+"_"+str(SLICING_PIXELS) + suffix)
            hkl.dump(train_df, filename_train, mode='w')



# In[11]:


def convertDatasetInList():
    global listPatientsDataset, dataset
    listPatientsDataset = {} # reset the list

    for patient_id in dataset.keys():
        listPatientsDataset[patient_id] = []
        for idx, pixels in enumerate(dataset[patient_id]["data"]):
            # print(np.array(pixels).shape)
            pixels = np.array(pixels).reshape(NUMBER_OF_IMAGE_PER_SECTION,M,N)
            ground_truth = dataset[patient_id]["ground_truth"][idx]
            label = dataset[patient_id]["label_class"][idx]
            listPatientsDataset[patient_id].append((patient_id, label, pixels, ground_truth))


# In[12]:


def divideDataForTrainAndTest():
    global listPatientsDataset, trainDatasetList

    for p_id in listPatientsDataset.keys():
        np.random.shuffle(listPatientsDataset[p_id])
        trainDatasetList.extend(listPatientsDataset[p_id])


# In[13]:


def prepareTraining():
    global trainDatasetList
    trainDatasetList = list() # reset
    # start the preparation for the training
    divideDataForTrainAndTest()
    tmp_df = pd.DataFrame(trainDatasetList, columns=['patient_id', 'label', 'pixels', 'ground_truth'])
    tmp_df['label_code'] = tmp_df.label.map({LABELS[0]:0, LABELS[1]:1, LABELS[2]:2, LABELS[3]:3})

    return tmp_df


# ## Main

if __name__ == '__main__':
    start = time.time()
    print("Initializing dataset...")
    initializeDataset()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))
