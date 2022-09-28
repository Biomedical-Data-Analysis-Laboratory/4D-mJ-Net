#!/usr/bin/env python
# coding: utf-8

################################################################################
# ### Import libraries
import cv2, time, glob, os, operator, random, math, argparse, json, multiprocessing
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import ndimage

dataset, listPatientsDataset, trainDatasetList = {}, {}, list()


# Util functions
################################################################################
def initializeLabels(patientIndex):
    global dataset
    dataset = dict()  # reset the dataset
    dataset[patientIndex] = dict()

    dataset[patientIndex]["data"] = list()
    dataset[patientIndex]["label_class"] = list()
    dataset[patientIndex]["ground_truth"] = list()


################################################################################
def getLabelledAreas(patientIndex, timeIndex):
    return cv2.imread(LABELED_IMAGES_FOLDER_LOCATION+IMAGE_PREFIX+patientIndex+"/"+timeIndex+GT_SUFFIX, cv2.IMREAD_GRAYSCALE)


################################################################################
# Function that return the slicing window from `img`, starting at pixels `startingX` and `startingY` with a width
# of `M` and height of `N`.
def getSlicingWindow(img, startingX, startingY, isgt=False):
    img = img[startingX:startingX + M, startingY:startingY + N]

    # check if there are any NaN elements
    if np.isnan(img).any():
        where = list(map(list, np.argwhere(np.isnan(img))))
        for w in where: img[w] = LABELS_REALVALUES[0]

    if isgt:
        for pxval in LABELS_REALVALUES:
            img = np.where(np.logical_and(
                img >= np.rint(pxval - (256 / 6)), img <= np.rint(pxval + (256 / 6))
            ), pxval, img)

        img = np.cast["float32"](img)  # cast the window into a float

    return img


################################################################################
def processTheWindow(realLabelledWindow, startingX, startingY, otherInforList, numBack, numSkip):
    valueClasses = dict()
    processTile = True  # use to skip the overlapping background tile (MEMORY ISSUES)

    if BINARY_CLASSIFICATION:  # JUST for 2 classes: core and the rest
        everything = realLabelledWindow >= 0
        binaryCoreMatrix = realLabelledWindow >= LABELS_THRESHOLDS[1]
        valueClasses[LABELS[1]] = sum(sum(binaryCoreMatrix))
        binaryEverything = everything ^ binaryCoreMatrix  # everything XOR core --> the other class
        valueClasses[LABELS[0]] = sum(sum(binaryEverything))

        # set a lower threshold for the core class
        if valueClasses[LABELS[1]] == 0: classToSet = LABELS[0]
        else:
            core_ratio = valueClasses[LABELS[1]] / sum(valueClasses.values())
            classToSet = LABELS[1] if not math.isnan(core_ratio) and core_ratio > 0.5 else LABELS[0]

    else:  # The normal four classes
        binaryBackgroundMatrix = realLabelledWindow >= LABELS_THRESHOLDS[0]
        binaryBrainMatrix = realLabelledWindow >= LABELS_THRESHOLDS[1]
        binaryPenumbraMatrix = realLabelledWindow >= LABELS_THRESHOLDS[2]
        binaryCoreMatrix = realLabelledWindow >= LABELS_THRESHOLDS[3]

        if not NEW_GROUNDTRUTH_VALUES:
            # extract the core area but not the brain area (= class 3)
            binaryCoreNoSkull = binaryBackgroundMatrix ^ binaryCoreMatrix  # background XOR core
            # extract the penumbra area but not the brain area (= class 2)
            binaryPenumbraNoSkull = binaryCoreMatrix ^ binaryPenumbraMatrix  # penumbra XOR
            # extract the brain area but not the background (= class 1)
            binaryBrainMatrixNoBackground = binaryBrainMatrix ^ binaryPenumbraMatrix  # brain XOR penumbra
            binaryBackground = binaryBackgroundMatrix
        else:
            binaryCoreNoSkull = binaryCoreMatrix
            binaryPenumbraNoSkull = binaryCoreMatrix ^ binaryPenumbraMatrix  # penumbra XOR core
            binaryBrainMatrixNoBackground = binaryBrainMatrix ^ binaryPenumbraMatrix  # brain XOR penumbra
            binaryBackground = binaryBackgroundMatrix ^ binaryBrainMatrix  # brain XOR background

        valueClasses[LABELS[0]] = sum(sum(binaryBackground))  # (= class 0)
        valueClasses[LABELS[1]] = sum(sum(binaryBrainMatrixNoBackground))
        valueClasses[LABELS[2]] = sum(sum(binaryPenumbraNoSkull))
        valueClasses[LABELS[3]] = sum(sum(binaryCoreMatrix))  # everything >= 230
        # set the window with just the four classes
        realLabelledWindow = (binaryBackground * LABELS_REALVALUES[0]) + \
                             (binaryCoreNoSkull * LABELS_REALVALUES[3]) + \
                             (binaryPenumbraNoSkull * LABELS_REALVALUES[2]) + \
                             (binaryBrainMatrixNoBackground * LABELS_REALVALUES[1])

        # the max of these values is the class to set for the binary class (Y)
        classToSet = max(valueClasses.items(), key=operator.itemgetter(1))[0]

    if classToSet == LABELS[0]:
        if SKIP_TILES:
            if startingY > 0 and startingY % N > 0:  # we are in a overlapping tile (Y dimension)
                jumps = int((startingY % N) / SLICING_PIXELS)
                for j in range(jumps):
                    prevTileY = (startingY - SLICING_PIXELS % N) - (j * SLICING_PIXELS)
                    if str(startingX) in otherInforList[0].keys() and str(prevTileY) in otherInforList[0][str(startingX)].keys():
                        if otherInforList[0][str(startingX)][str(prevTileY)]["label_class"] == classToSet:
                            numSkip += 1
                            processTile = False
                            break
            if not processTile and startingX % M > 0:  # we are in a overlapping tile (X dimension)
                jumps = int((startingX % N) / SLICING_PIXELS)
                for j in range(jumps):
                    prevTileX = (startingX - SLICING_PIXELS % N) - (j * SLICING_PIXELS)
                    if str(prevTileX) in otherInforList[0].keys() and str(startingY) in otherInforList[0][str(prevTileX)].keys():
                        if otherInforList[0][str(prevTileX)][str(startingY)]["label_class"] == classToSet:
                            numSkip += 1
                            processTile = False
                            break
        if processTile: numBack += 1

    return realLabelledWindow, classToSet, processTile, numBack, numSkip


################################################################################
# Function for inserting inside `dataset` the pixel areas (slicing windows) found with
# a slicing area approach (= start from point 0,0 it takes the areas `MxN` and then move on the right or on the bottom
# by a pre-fixed number of pixels = `SLICING_PIXELS`) and the corresponding area in the images inside the same folder,
# which are the registered images of the same section of the brain in different time.
def fillDatasetOverTime(relativePath, patientIndex, timeFolder, infor_file):
    listOfSeries = list()

    numReplication, numRep = 1, 1
    if DATA_AUGMENTATION: numRep = 6
    numBack, numBrain, numPenumbra, numCore, numSkip = 0, 0, 0, 0, 0
    startingX, startingY = 0, 0
    pixelsList, otherInforList = list(), list()

    sliceIndex = timeFolder.replace(SAVE_REGISTERED_FOLDER+relativePath, '').replace("/", "")
    if len(sliceIndex)==1: sliceIndex="0"+sliceIndex

    labelledMatrix = getLabelledAreas(patientIndex, sliceIndex)

    imagesDict = {}  # faster access to the images
    if not SEQUENCE_DATASET:
        for imagename in np.sort(glob.glob(timeFolder+"*"+IMAGE_SUFFIX)):  # sort the images !
            filename = imagename.replace(timeFolder, '')
            # don't take the first image (the manually annotated one)
            if "OLDPREPROC_PATIENTS/" in SAVE_REGISTERED_FOLDER and filename == "01"+IMAGE_SUFFIX: continue
            image = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
            imagesDict[filename] = image

    for _ in range(numRep):
        pixelsList.append(dict())
        otherInforList.append(dict())

    start = time.time()
    while True:
        realLabelledWindow = getSlicingWindow(labelledMatrix, startingX, startingY, isgt=True)
        # process the window; return the new labeled window and various flags
        realLabelledWindow, classToSet, processTile, numBack, numSkip = processTheWindow(realLabelledWindow, startingX, startingY, otherInforList, numBack, numSkip)

        if BINARY_CLASSIFICATION:
            if classToSet != LABELS[0]:
                numReplication = 6 if DATA_AUGMENTATION else 1
                numCore += numReplication

            # if TILE_DIVISION == 1 and DATA_AUGMENTATION: numReplication = 6
        else:
            if classToSet == LABELS[1]: numBrain += 1
            elif classToSet == LABELS[2]: numPenumbra += 1
            elif classToSet == LABELS[3]:
                numReplication = 6 if DATA_AUGMENTATION else 1
                numCore += numReplication

        if processTile:
            for data_aug_idx in range(numReplication):  # start from 0
                # loop the SORTED images !
                for image_idx, imagename in enumerate(np.sort(glob.glob(timeFolder+"*"+IMAGE_SUFFIX))):
                    if str(startingX) not in pixelsList[data_aug_idx].keys(): pixelsList[data_aug_idx][str(startingX)] = dict()
                    if str(startingX) not in otherInforList[data_aug_idx].keys(): otherInforList[data_aug_idx][str(startingX)] = dict()
                    if str(startingY) not in pixelsList[data_aug_idx][str(startingX)].keys(): pixelsList[data_aug_idx][str(startingX)][str(startingY)] = dict()
                    if str(startingY) not in otherInforList[data_aug_idx][str(startingX)].keys(): otherInforList[data_aug_idx][str(startingX)][str(startingY)] = dict()

                    realLabelledWindowToAdd = realLabelledWindow
                    if data_aug_idx == 1: realLabelledWindowToAdd = np.rot90(realLabelledWindowToAdd)
                    elif data_aug_idx == 2: realLabelledWindowToAdd = np.rot90(realLabelledWindowToAdd,2)
                    elif data_aug_idx == 3: realLabelledWindowToAdd = np.rot90(realLabelledWindowToAdd,3)
                    elif data_aug_idx == 4: realLabelledWindowToAdd = np.flipud(realLabelledWindowToAdd)
                    elif data_aug_idx == 5: realLabelledWindowToAdd = np.fliplr(realLabelledWindowToAdd)

                    # process and save the pixels only if we are NOT creating a sequence dataset
                    if not SEQUENCE_DATASET:
                        filename = imagename.replace(timeFolder, '')
                        # don't take the first image (the manually annotated one)
                        if "OLDPREPROC_PATIENTS/" in SAVE_REGISTERED_FOLDER and filename == "01"+IMAGE_SUFFIX: continue

                        image = imagesDict[filename]
                        slicingWindow = getSlicingWindow(image, startingX, startingY)

                        if data_aug_idx==1: slicingWindow = np.rot90(slicingWindow)  # rotate 90 degree counterclockwise
                        elif data_aug_idx==2: slicingWindow = np.rot90(slicingWindow,2)  # rotate 180 degree counterclockwise
                        elif data_aug_idx==3: slicingWindow = np.rot90(slicingWindow,3)  # rotate 270 degree counterclockwise
                        elif data_aug_idx==4: slicingWindow = np.flipud(slicingWindow)  # flip the matrix up/down
                        elif data_aug_idx==5: slicingWindow = np.fliplr(slicingWindow)  # flip the matrix left/right

                        if image_idx not in pixelsList[data_aug_idx][str(startingX)][str(startingY)].keys(): pixelsList[data_aug_idx][str(startingX)][str(startingY)][image_idx] = list()
                        pixelsList[data_aug_idx][str(startingX)][str(startingY)][image_idx] = slicingWindow

                # if we are processing for the sequence dataset, save the path for the ground truth
                if SEQUENCE_DATASET:
                    otherInforList[data_aug_idx][str(startingX)][str(startingY)]["ground_truth"] = LABELED_IMAGES_FOLDER_LOCATION+IMAGE_PREFIX+patientIndex+"/"+sliceIndex+GT_SUFFIX
                    if HASDAYFOLDER: otherInforList[data_aug_idx][str(startingX)][str(startingY)]["mask"] = MASKS_IMAGES_FOLDER_LOCATION + IMAGE_PREFIX + patientIndex + "/" + sliceIndex + IMAGE_SUFFIX
                    otherInforList[data_aug_idx][str(startingX)][str(startingY)]["pixels"] = timeFolder

                    if HASDAYFOLDER:  # SUS2020 dataset
                        for dayfolder in np.sort(glob.glob(PM_FOLDER + IMAGE_PREFIX + patientIndex +"/*/")):
                            # if the folder contains the correct number of subfolders
                            n_fold = 7
                            if "20_" in patientIndex or "21_" in patientIndex or "22_" in patientIndex or "23_" in patientIndex: n_fold=5

                            if len(glob.glob(dayfolder+"*/"))>=n_fold:
                                pmlist = ["CBF", "CBV", "TTP", "TMAX", "MIP"]

                                for subdayfolder in glob.glob(dayfolder+"*/"):
                                    for pm in pmlist:
                                        if pm in subdayfolder: otherInforList[data_aug_idx][str(startingX)][str(startingY)][pm] = subdayfolder+sliceIndex+".png"
                    else:  # ISLES2018 dataset
                        if len(glob.glob(PM_FOLDER + IMAGE_PREFIX + patientIndex +"/*/")) >= 6:
                            pmlist = ["CBF", "CBV", "MTT", "Tmax"]
                            for listpms in glob.glob(PM_FOLDER + IMAGE_PREFIX + patientIndex + "/*/"):
                                for pm in pmlist:
                                    if pm in listpms: otherInforList[data_aug_idx][str(startingX)][str(startingY)][pm.upper()] = listpms+sliceIndex+".tiff"

                else: otherInforList[data_aug_idx][str(startingX)][str(startingY)]["ground_truth"] = realLabelledWindowToAdd

                otherInforList[data_aug_idx][str(startingX)][str(startingY)]["label_class"] = classToSet
                otherInforList[data_aug_idx][str(startingX)][str(startingY)]["x_y"] = (startingX, startingY)
                otherInforList[data_aug_idx][str(startingX)][str(startingY)]["data_aug_idx"] = data_aug_idx
                otherInforList[data_aug_idx][str(startingX)][str(startingY)]["timeIndex"] = imagename.replace(timeFolder, '').replace(IMAGE_SUFFIX,"")
                otherInforList[data_aug_idx][str(startingX)][str(startingY)]["sliceIndex"] = sliceIndex
                if HASDAYFOLDER: otherInforList[data_aug_idx][str(startingX)][str(startingY)]["severity"] = patientIndex.split("_")[0].replace("20", "00").replace("21", "01").replace("22", "02").replace("23", "03")

        # if we reach the end of the image, break the while loop.
        if startingX >= IMAGE_WIDTH - M and startingY >= IMAGE_HEIGHT - N: break
        # check for M == WIDTH & N == HEIGHT
        if M == IMAGE_WIDTH and N == IMAGE_HEIGHT: break
        # going to the next slicingWindow
        if startingY<IMAGE_HEIGHT-N: startingY += SLICING_PIXELS
        else:
            if startingX<IMAGE_WIDTH-M:
                startingY = 0
                startingX += SLICING_PIXELS

    print("\t Processed while loop in {}s.".format(round(time.time()-start,3)))
    start = time.time()
    if VERBOSE:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\t\t Background: {0}".format(numBack))
        if not BINARY_CLASSIFICATION:
            print("\t\t Brain: {0}".format(numBrain))
            print("\t\t Penumbra: {0}".format(numPenumbra))
        print("\t\t Core: {0}".format(numCore))
        print("\t\t SKIP: {0}".format(numSkip))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

    axis = 2
    if ORIGINAL_SHAPE: axis = 0
    for d in range(0, len(pixelsList)):
        for x in pixelsList[d].keys():
            for y in pixelsList[d][x].keys():

                if SEQUENCE_DATASET:
                    pixels_zoom = otherInforList[d][x][y]["pixels"]
                    if HASDAYFOLDER:
                        cbf = otherInforList[d][x][y]["CBF"] if "CBF" in otherInforList[d][x][y].keys() else ""
                        cbv = otherInforList[d][x][y]["CBV"] if "CBV" in otherInforList[d][x][y].keys() else ""
                        ttp = otherInforList[d][x][y]["TTP"] if "TTP" in otherInforList[d][x][y].keys() else ""
                        tmax = otherInforList[d][x][y]["TMAX"] if "TMAX" in otherInforList[d][x][y].keys() else ""
                        mip = otherInforList[d][x][y]["MIP"] if "MIP" in otherInforList[d][x][y].keys() else ""
                    else:
                        cbf = otherInforList[d][x][y]["CBF"] if "CBF" in otherInforList[d][x][y].keys() else ""
                        cbv = otherInforList[d][x][y]["CBV"] if "CBV" in otherInforList[d][x][y].keys() else ""
                        mtt = otherInforList[d][x][y]["MTT"] if "MTT" in otherInforList[d][x][y].keys() else ""
                        tmax = otherInforList[d][x][y]["TMAX"] if "TMAX" in otherInforList[d][x][y].keys() else ""

                else:
                    if ORIGINAL_SHAPE: totalVol = np.empty((1,M,N))
                    else: totalVol = np.empty((M,N,1))

                    for z in sorted(pixelsList[d][x][y].keys()):
                        tmp_pix = np.array(pixelsList[d][x][y][z])

                        if ORIGINAL_SHAPE: tmp_pix = tmp_pix.reshape(1, tmp_pix.shape[0], tmp_pix.shape[1])
                        else: tmp_pix = tmp_pix.reshape(tmp_pix.shape[0], tmp_pix.shape[1], 1)
                        totalVol = np.append(totalVol, tmp_pix, axis=axis)

                    totalVol = np.delete(totalVol,0,axis=axis)  # remove the first element (generate by np.empty)

                    # convert the pixels in a (M,N,30) shape (or (30,M,N) if ORIGINAL_SHAPE==True)
                    zoom_val = NUMBER_OF_IMAGE_PER_SECTION/totalVol.shape[axis]

                    if totalVol.shape[axis] > NUMBER_OF_IMAGE_PER_SECTION:
                        zoom_val = totalVol.shape[axis]/NUMBER_OF_IMAGE_PER_SECTION

                    if ORIGINAL_SHAPE: pixels_zoom = ndimage.zoom(totalVol,[zoom_val,1,1],output=np.uint8)
                    else: pixels_zoom = ndimage.zoom(totalVol,[1,1,zoom_val],output=np.uint8)

                # # USE THIS TO CHECK THE VALIDITIY OF THE INTERPOLATION
                # print(pixels_zoom.shape)
                # for z in range(0,pixels_zoom.shape[0]):
                #     print(ROOT_PATH+"Test/img_{0}_{1}_{2}_{3}.png".format(d,x,y,z))
                #     cv2.imwrite(ROOT_PATH+"Test/img_{0}_{1}_{2}_{3}.png".format(d,x,y,z),  pixels_zoom[z,:,:])
                #     if totalVol.shape[0] > z:
                #         print(ROOT_PATH+"Test/origimg_{0}_{1}_{2}_{3}.png".format(d,x,y,z))
                #         cv2.imwrite(ROOT_PATH+"Test/origimg_{0}_{1}_{2}_{3}.png".format(d,x,y,z), totalVol[z,:,:])

                label = otherInforList[d][x][y]["label_class"]
                gt = otherInforList[d][x][y]["ground_truth"]
                if HASDAYFOLDER: mask = otherInforList[d][x][y]["mask"]
                x_y = otherInforList[d][x][y]["x_y"]
                data_aug_idx = otherInforList[d][x][y]["data_aug_idx"]
                timeIndex = otherInforList[d][x][y]["timeIndex"]
                if HASDAYFOLDER: severity = otherInforList[d][x][y]["severity"]

                tmp_COLUMNS = list(filter(lambda col:col != 'label_code', COLUMNS))

                if HASDAYFOLDER:
                    nihss = infor_file.nihss[IMAGE_PREFIX+patientIndex]
                    age = int(infor_file.age[IMAGE_PREFIX+patientIndex][:-1])
                    gender = 0 if infor_file.gender[IMAGE_PREFIX+patientIndex] == "M" else 1

                    tmp_df = pd.DataFrame(np.array([[patientIndex, label, pixels_zoom, cbf, cbv, ttp, tmax, mip, nihss, gt, mask,
                                                     x_y, data_aug_idx, timeIndex, sliceIndex, severity, age, gender]], dtype=object),
                                          columns=tmp_COLUMNS)
                else:
                    tmp_df = pd.DataFrame(np.array([[patientIndex, label, pixels_zoom, cbf, cbv, mtt, tmax, gt,
                                                     x_y, data_aug_idx, timeIndex, sliceIndex]]), columns=tmp_COLUMNS)

                if BINARY_CLASSIFICATION: tmp_df['label_code'] = tmp_df.label.map({LABELS[0]:0, LABELS[1]:1})
                else: tmp_df['label_code'] = tmp_df.label.map({LABELS[0]:0, LABELS[1]:1, LABELS[2]:2, LABELS[3]:3})
                listOfSeries.append(tmp_df)

    print("\t Processed fillDatasetOverTime in {}s.".format(round(time.time()-start,3)))
    return listOfSeries


################################################################################
# Function that initialize the dataset: for each subfolder of the patient (section of the brain),
# it call the `fillDataset` function to get the pixels, save into the dataset and analyze them later.
def initializeDataset():
    patientFolders = glob.glob(SAVE_REGISTERED_FOLDER+"*/")
    suffix_filename = "_"+str(SLICING_PIXELS)+"_"+str(M)+"x"+str(N)

    if ONE_TIME_POINT>0:
        timeIndex = str(ONE_TIME_POINT)
        if len(timeIndex)==1: timeIndex="0"+timeIndex
        suffix_filename += ("_"+timeIndex)

    infor_file = ""
    if HASDAYFOLDER: infor_file = pd.read_csv("../nihss_score.csv",index_col=0,sep=";")
    lpf = len(patientFolders)
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(runSingleDataframe, list(zip(range(0, lpf), patientFolders, [suffix_filename]*lpf, [infor_file]*lpf, [lpf]*lpf)))

    # for numFold, patientFolder in enumerate(patientFolders):  # for each patient
    #     runSingleDataframe(numFold, patientFolder, suffix_filename, infor_file)


################################################################################
def runSingleDataframe(numFold, patientFolder, suffix_filename, infor_file, lpf):
        train_df = pd.DataFrame(columns=COLUMNS)  # reset the dataframe

        relativePath = patientFolder.replace(SAVE_REGISTERED_FOLDER, '')
        patientIndex = relativePath.replace(IMAGE_PREFIX, "").replace("/", "")

        filename_train = SCRIPT_PATH+"patient"+str(patientIndex)+suffix_filename+".pkl"

        if os.path.isfile(filename_train):
            print("File {} already exist, continue...".format(filename_train))
            return

        subfolders = np.sort(glob.glob(patientFolder+"*/"))
        print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold+1, lpf, relativePath))
        # if the manual annotation folder exists
        if os.path.isdir(LABELED_IMAGES_FOLDER_LOCATION+IMAGE_PREFIX+patientIndex+"/"):
            for count, timeFolder in enumerate(subfolders):  # for each slicing time
                initializeLabels(patientIndex)
                print("\t Analyzing subfolder {0}".format(timeFolder.replace(SAVE_REGISTERED_FOLDER, '').replace(relativePath, '')))
                start = time.time()
                tmp_df = fillDatasetOverTime(relativePath, patientIndex, timeFolder, infor_file)
                train_df = train_df.append(tmp_df, ignore_index=True, sort=True)
                print("\t Processed {0}/{1} subfolders in {2}s.".format(count+1, len(subfolders), round(time.time()-start, 3)))
                if VERBOSE: print("Train shape: ", train_df.shape)

            ################################################################################
            if VERBOSE: print("Saving TRAIN dataframe for patient {1} in {0}...".format(filename_train, str(patientIndex)))

            # save pickle and hickle version
            f = open(filename_train, 'wb')
            pkl.dump(train_df, f)


################################################################################
# Get the setting file and set the variables
def getSettingFile(filename):
    global setting
    # the path of the setting file start from the current working directory
    with open(os.path.join(os.getcwd(), filename)) as f: setting = json.load(f)
    if VERBOSE: print("Load setting file: {}".format(filename))


################################################################################
# Set the setting from the file
def setSettings():
    global DATASET_NAME, ROOT_PATH, SCRIPT_PATH, SAVE_REGISTERED_FOLDER, LABELED_IMAGES_FOLDER_LOCATION, IMAGE_PREFIX
    global IMAGE_SUFFIX, GT_SUFFIX, NUMBER_OF_IMAGE_PER_SECTION, IMAGE_WIDTH, IMAGE_HEIGHT
    global BINARY_CLASSIFICATION, LABELS, LABELS_THRESHOLDS, LABELS_REALVALUES, TILE_DIVISION, SEQUENCE_DATASET
    global SKIP_TILES, ORIGINAL_SHAPE, DATA_AUGMENTATION, ONE_TIME_POINT, COLUMNS, setting
    global NEW_GROUNDTRUTH_VALUES, M, N, SLICING_PIXELS, MASKS_IMAGES_FOLDER_LOCATION, PM_FOLDER

    DATASET_NAME = setting["DATASET_NAME"]
    ROOT_PATH = setting["ROOT_PATH"]
    SCRIPT_PATH = setting["SCRIPT_PATH"]
    SAVE_REGISTERED_FOLDER = setting["SAVE_REGISTERED_FOLDER"]
    PM_FOLDER = setting["PM_FOLDER"]
    LABELED_IMAGES_FOLDER_LOCATION = setting["LABELED_IMAGES_FOLDER_LOCATION"]
    MASKS_IMAGES_FOLDER_LOCATION = setting["MASKS_IMAGES_FOLDER_LOCATION"]
    IMAGE_PREFIX = setting["IMAGE_PREFIX"]
    IMAGE_SUFFIX = setting["IMAGE_SUFFIX"]
    GT_SUFFIX = setting["GT_SUFFIX"]
    NUMBER_OF_IMAGE_PER_SECTION = setting["NUMBER_OF_IMAGE_PER_SECTION"]
    IMAGE_WIDTH = setting["IMAGE_WIDTH"]
    IMAGE_HEIGHT = setting["IMAGE_HEIGHT"]
    BINARY_CLASSIFICATION = setting["BINARY_CLASSIFICATION"]
    LABELS = setting["LABELS"]
    LABELS_THRESHOLDS = setting["LABELS_THRESHOLDS"]
    LABELS_REALVALUES = setting["LABELS_REALVALUES"]
    TILE_DIVISION = setting["TILE_DIVISION"]
    # create a dataset compatible with the Keras Sequence class https://keras.io/api/utils/python_utils/
    SEQUENCE_DATASET = setting["SEQUENCE_DATASET"]
    SKIP_TILES = setting["SKIP_TILES"]  # skip the tiles?
    ORIGINAL_SHAPE = setting["ORIGINAL_SHAPE"]  # the one from the master thesis
    DATA_AUGMENTATION = setting["DATA_AUGMENTATION"]  # use data augmentation?
    ONE_TIME_POINT = setting["ONE_TIME_POINT"]  # -1 if you don't want to use it otherwise select 1 timepoint to extract
    COLUMNS = setting["COLUMNS"]
    NEW_GROUNDTRUTH_VALUES = setting["NEW_GROUNDTRUTH_VALUES"]  # flag to use the new GT values

    ################################################################################
    M, N = int(IMAGE_WIDTH/TILE_DIVISION), int(IMAGE_HEIGHT/TILE_DIVISION)
    SLICING_PIXELS = int(M/4)  # USE ALWAYS M/4

    if not os.path.isdir(SCRIPT_PATH): os.mkdir(SCRIPT_PATH)

    if NEW_GROUNDTRUTH_VALUES and not BINARY_CLASSIFICATION:
        LABELS_THRESHOLDS = [0, 70, 155, 230]  # [250, 0 , 30, 100]
        LABELS_REALVALUES = [0, 85, 170, 255]


################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    global DATASET_NAME, ROOT_PATH, SCRIPT_PATH, SAVE_REGISTERED_FOLDER, LABELED_IMAGES_FOLDER_LOCATION, IMAGE_PREFIX
    global IMAGE_SUFFIX, GT_SUFFIX, NUMBER_OF_IMAGE_PER_SECTION, IMAGE_WIDTH, IMAGE_HEIGHT
    global BINARY_CLASSIFICATION, LABELS, LABELS_THRESHOLDS, LABELS_REALVALUES, TILE_DIVISION, SEQUENCE_DATASET
    global SKIP_TILES, ORIGINAL_SHAPE, DATA_AUGMENTATION, ONE_TIME_POINT, COLUMNS, PM_FOLDER
    global NEW_GROUNDTRUTH_VALUES, M, N, SLICING_PIXELS, MASKS_IMAGES_FOLDER_LOCATION, setting

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("sname", help="Select the setting filename")
    parser.add_argument("-d", "--dayfold", help="Flag for having a dayfolder in the parametric maps folder", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    HASDAYFOLDER = args.dayfold
    getSettingFile(args.sname)
    setSettings()

    start = time.time()
    print("Initializing dataset...")
    initializeDataset()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))
