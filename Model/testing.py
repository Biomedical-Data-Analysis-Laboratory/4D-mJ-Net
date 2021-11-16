# Run the testing function, save the images ..
import cv2
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import time
import warnings
import pickle as pkl
import seaborn as sns
import tensorflow.keras.backend as K
from scipy import ndimage
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from Model import constants, training
from Utils import general_utils, dataset_utils, sequence_utils, metrics, model_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Get the labeled image processed (= GT)
def getCheckImageProcessed(nn, p_id, idx):
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    # get the label image only if the path is set
    if nn.labeledImagesFolder != "":
        filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + idx + constants.SUFFIX_IMG
        if not os.path.exists(filename):
            print("[WARNING] - {0} does NOT exists, try another...".format(filename))
            filename = nn.labeledImagesFolder + constants.PREFIX_IMAGES + p_id + "/" + p_id + idx + constants.SUFFIX_IMG
            assert os.path.exists(filename), "[ERROR] - {0} does NOT exist".format(filename)

        checkImageProcessed = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        if len(checkImageProcessed.shape)==3: checkImageProcessed=cv2.cvtColor(checkImageProcessed, cv2.COLOR_BGR2GRAY)
        assert len(checkImageProcessed.shape)==2, "The GT image shape should be 2."
    return checkImageProcessed


################################################################################
# Predict the model based on the input
def predictFromModel(nn, x_input):
    return nn.model.predict(x=x_input, batch_size=1, use_multiprocessing=nn.mp_in_nn)


################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(nn, p_id):
    suffix = general_utils.getSuffix()  # es == "_4_16x16"

    suffix_filename = ".pkl"
    if nn.use_hickle: suffix_filename = ".hkl"
    filename_test = nn.datasetFolder + constants.DATASET_PREFIX + str(p_id) + suffix + suffix_filename

    relativePatientFolder = constants.getPrefixImages() + str(p_id) + "/"
    relativePatientFolderHeatMap = relativePatientFolder + "HEATMAP/"
    relativePatientFolderGT = relativePatientFolder + "GT/"
    relativePatientFolderTMP = relativePatientFolder + "TMP/"
    patientFolder = nn.patientsFolder+relativePatientFolder

    filename_saveImageFolder = nn.saveImagesFolder+nn.experimentID+"__"+nn.getNNID()+suffix
    # create the related folders
    general_utils.createDir(filename_saveImageFolder)
    for subpath in [relativePatientFolder,relativePatientFolderHeatMap,relativePatientFolderGT,relativePatientFolderTMP]:
        general_utils.createDir(filename_saveImageFolder+"/"+subpath)

    prefix = nn.experimentID + constants.suffix_partial_weights + nn.getNNID() + suffix + "/"
    subpatientFolder = prefix+relativePatientFolder
    patientFolderHeatMap = prefix+relativePatientFolderHeatMap
    patientFolderGT = prefix+relativePatientFolderGT
    patientFolderTMP = prefix+relativePatientFolderTMP

    # for all the slice folders in patientFolder
    for subfolder in glob.glob(patientFolder+"*/"):
        # Predict the images
        if constants.getUSE_PM(): predictImagesFromParametricMaps(nn, subfolder, p_id, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)
        else:
            if constants.getIsISLES2018(): predictImage(nn, subfolder, p_id, patientFolder, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)
            else: predictImage(nn, subfolder, p_id, patientFolder, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)


################################################################################
def predictImage(nn, subfolder, p_id, patientFolder, relativePatientFolder, relativePatientFolderHeatMap,
                 relativepatientFolderGT, relativePatientFolderTMP, filename_test):
    """
    Generate a SINGLE image for the patient and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - relativePatientFolder         : relative name of the patient folder
    - relativePatientFolderHeatMap  : relative name of the patient heatmap folder
    - relativepatientFolderGT       : relative name of the patient gt folder
    - relativePatientFolderTMP      : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """

    imagesDict = {}  # faster access to the images
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))

    idx = general_utils.getStringFromIndex(subfolder.replace(patientFolder, '').replace("/", ""))  # image index

    # remove the old logs.
    logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    if os.path.isfile(logsName): os.remove(logsName)

    if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    checkImageProcessed = getCheckImageProcessed(nn, p_id, idx)
    # binary_mask = np.zeros(shape=(constants.getM(), constants.getN()))
    binary_mask = checkImageProcessed != constants.PIXELVALUES[0]

    # folders = [subfolder]
    # if nn.is4DModel and nn.n_slices>1: folders = model_utils.getPrevNextFolder(subfolder, idx)
    # for fold in folders:
    #     # get the images in a dictionary
    #     for imagename in np.sort(glob.glob(fold +"*" + constants.SUFFIX_IMG)):  # sort the images !
    #         filename = imagename.replace(fold, '')
    #         if not nn.supervised or nn.patientsFolder!="OLDPREPROC_PATIENTS/": imagesDict[imagename] = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
    #         else:
    #             if filename != "01"+ constants.SUFFIX_IMG: imagesDict[imagename] = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)

    # Portion for the prediction of the image
    if constants.get3DFlag()!= "":
        assert os.path.exists(filename_test), "[ERROR] - File {} does NOT exist".format(filename_test)

        test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
        # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
        test_df = test_df[test_df.data_aug_idx==0]
        print(test_df.shape)
        test_df = test_df[test_df.timeIndex==idx]  # todo: why timeindex?
        print(test_df.shape)
        imagePredicted = generateTimeImagesAndConsensus(nn, test_df, relativepatientFolderGT, relativePatientFolderTMP, idx)
    else:  # usual behaviour
        while True:
            test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
            test_df = test_df[test_df.x_y == (startingX, startingY)]
            test_df = test_df[test_df.sliceIndex == idx]
            row = test_df[test_df.data_aug_idx == 0]
            # Control that the analyzed row is == 1
            assert len(row) == 1, "The length of the row to analyze should be 1."
            X = model_utils.getCorrectXForInputModel(nn, subfolder, row, batchIndex=0, batch_length=1)

            imagePredicted, categoricalImage = generate2DImage(nn, X, (startingX,startingY), imagePredicted, categoricalImage, binary_mask)

            # if we reach the end of the image, break the while loop.
            if startingX>=constants.IMAGE_WIDTH-constants.getM() and startingY>=constants.IMAGE_HEIGHT-constants.getN(): break

            # going to the next slicingWindow
            if startingY< constants.IMAGE_HEIGHT- constants.getN(): startingY+= constants.getN()
            else:
                if startingX< constants.IMAGE_WIDTH:
                    startingY=0
                    startingX+= constants.getM()

    # save the image
    saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage, relativePatientFolderHeatMap,
              relativepatientFolderGT, relativePatientFolderTMP,checkImageProcessed)


################################################################################
# Function for predicting a brain slice based on the parametric maps
def predictImagesFromParametricMaps(nn, subfolder, p_id, relativePatientFolder, relativePatientFolderHeatMap, relativepatientFolderGT, relativePatientFolderTMP, filename_test):
    """
    Generate ALL the images for the patient using the PMs and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - relativePatientFolder         : relative name of the patient folder
    - relativePatientFolderHeatMap  : relative name of the patient heatmap folder
    - relativepatientFolderGT      : relative name of the patient gt folder
    - relativePatientFolderTMP      : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """

    # if the patient folder contains the correct number of subfolders
    # ATTENTION: careful here...
    n_fold = 7
    if constants.getIsISLES2018(): n_fold = 5

    if len(glob.glob(subfolder+"*/"))>=n_fold:
        for idx in glob.glob(subfolder+"/CBF/*"):
            idx = general_utils.getStringFromIndex(idx.replace(subfolder, '').replace("/CBF/", ""))  # image index
            if constants.getIsISLES2018(): idx = idx.replace(".tiff","")
            else: idx = idx.replace(".png","")
            # remove the old logs.
            logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
            if os.path.isfile(logsName): os.remove(logsName)
    
            # if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))

            checkImageProcessed = getCheckImageProcessed(nn, str(p_id), idx)

            assert os.path.exists(filename_test), "[ERROR] - File {0} does NOT exist".format(filename_test)
    
            # get the pandas dataframe
            test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
            # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
            test_df = test_df[test_df.data_aug_idx == 0]
            test_df = test_df[test_df.sliceIndex == idx]
    
            imagePredicted, categoricalImage = generateImageFromParametricMaps(nn, test_df)
    
            # save the image
            saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage, relativePatientFolderHeatMap,
                      relativepatientFolderGT, relativePatientFolderTMP, checkImageProcessed)


################################################################################
# Util function to save image
def saveImage(nn, relativePatientFolder, idx, imagePredicted, categoricalImage, relativePatientFolderHeatMap,
              relativepatientFolderGT, relativePatientFolderTMP, checkImageProcessed):

    if nn.save_images:
        # save the image predicted in the specific folder
        cv2.imwrite(nn.saveImagesFolder+relativePatientFolder+idx+".png", imagePredicted)
        # create and save the HEATMAP only if we are using softmax activation
        if constants.getTO_CATEG() and nn.labeledImagesFolder!="":
            p_idx, c_idx = 2,3
            if constants.N_CLASSES==3: p_idx, c_idx = 1, 2
            elif constants.N_CLASSES==2: c_idx = 1

            checkImageProcessed_rgb = cv2.cvtColor(checkImageProcessed, cv2.COLOR_GRAY2RGB)
            if constants.N_CLASSES >= 3:
                heatmap_img_p = cv2.convertScaleAbs(categoricalImage[:, :, p_idx] * 255)
                heatmap_img_p = cv2.applyColorMap(heatmap_img_p, cv2.COLORMAP_JET)
                blend_p = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_p, 0.5, 0.0)
                cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_penumbra.png", blend_p)
            heatmap_img_c = cv2.convertScaleAbs(categoricalImage[:, :, c_idx] * 255)
            heatmap_img_c = cv2.applyColorMap(heatmap_img_c, cv2.COLORMAP_JET)
            blend_c = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_c, 0.5, 0.0)
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_core.png", blend_c)

        # Save the ground truth and the contours
        if constants.get3DFlag()== "" and nn.labeledImagesFolder!="":
            # save the GT
            cv2.imwrite(nn.saveImagesFolder + relativepatientFolderGT + idx + constants.SUFFIX_IMG, checkImageProcessed)

            imagePredicted = cv2.cvtColor(np.uint8(imagePredicted),cv2.COLOR_GRAY2RGB)  # back to rgb
            if constants.N_CLASSES >= 3:
                _, penumbra_mask = cv2.threshold(checkImageProcessed, 85, constants.PIXELVALUES[-2], cv2.THRESH_BINARY)
                penumbra_cnt, _ = cv2.findContours(penumbra_mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                imagePredicted = cv2.drawContours(imagePredicted, penumbra_cnt, -1, (255,0,0), 2)
            _, core_mask = cv2.threshold(checkImageProcessed, constants.PIXELVALUES[-2], constants.PIXELVALUES[-1], cv2.THRESH_BINARY)
            core_cnt, _ = cv2.findContours(core_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imagePredicted = cv2.drawContours(imagePredicted, core_cnt, -1, (0,0,255), 2)
            # save the GT image with predicted contours
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, penumbra_area, 0.5, 0.0)
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, core_area, 0.5, 0.0)
            cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+idx+".png",imagePredicted)


################################################################################
# Helpful function that return the 2D image from the pixel and the starting coordinates
def generate2DImage(nn,pixels,startingXY,imgPredicted,categoricalImage,binary_mask):
    """
    Generate a 2D image from the test_df

    Input parameters:
    - nn                    : NeuralNetwork class
    - pixels                : pixel in a numpy array
    - startingXY            : (x,y) coordinates
    - imgPredicted          : the predicted image
    - categoricalImage      : the categorical image predicted
    - binary_mask           : the binary mask containing the skull

    Return:
    - imgPredicted          : the predicted image
    """
    x, y = startingXY
    # swp_orig contain only the prediction for the last step
    swp_orig = predictFromModel(nn, pixels)[0]
    if nn.save_images and constants.getTO_CATEG(): categoricalImage[x:x+constants.getM(),y:y+constants.getN()]=swp_orig

    # convert the categorical into a single array for removing some uncertain predictions
    if constants.getTO_CATEG(): slicingWindowPredicted = K.eval((K.argmax(swp_orig) * 255) / (constants.N_CLASSES - 1))
    else: slicingWindowPredicted = swp_orig * 255
    # save the predicted images
    if nn.save_images:
        if not constants.getIsISLES2018():
            # Remove the parts already classified by the model
            binary_mask = np.array(binary_mask, dtype=np.float)
            # force all the predictions to be inside the binary mask defined by the GT
            slicingWindowPredicted *= binary_mask[x:x+constants.getM(),y:y+constants.getN()]

            overlapping_pred = np.array(slicingWindowPredicted>0,dtype=np.float)
            overlapping_pred *= 85.
            binary_mask *= 85.  # multiply the binary mask for the brain pixel value
            # add the brain to the prediction window
            slicingWindowPredicted += (binary_mask[x:x+constants.getM(),y:y+constants.getN()]-overlapping_pred)
        imgPredicted[x:x+constants.getM(),y:y+constants.getN()]=slicingWindowPredicted
    return imgPredicted, categoricalImage


################################################################################
def generateTimeImagesAndConsensus(nn, test_df, relativePatientFolderTMP, idx):
    """
    Generate the image from the 3D sequence of time index (create also these images) with a consensus

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - relativePatientFolderTMP  : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - imagePredicted        : the predicted image
    """

    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    arrayTimeIndexImages = dict()

    for test_row in test_df.itertuples():  # for every rows of the same image
        if str(test_row.timeIndex) not in arrayTimeIndexImages.keys(): arrayTimeIndexImages[str(test_row.timeIndex)] = np.zeros(shape=(
        constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)
        if constants.get3DFlag() == "": test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2], 1)
        else: test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2])
        arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage = generate2DImage(nn, test_row.pixels, test_row.x_y, arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage, checkImageProcessed)

    if nn.save_images:              # remove one class from the ground truth
        if constants.N_CLASSES==3: checkImageProcessed[checkImageProcessed == 85] = constants.PIXELVALUES[0]
        cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP +"orig_" + idx + constants.SUFFIX_IMG, checkImageProcessed)

        for tidx in arrayTimeIndexImages.keys():
            curr_image = arrayTimeIndexImages[tidx]
            # save the images
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP + idx +"_" + general_utils.getStringFromIndex(tidx) + constants.SUFFIX_IMG, curr_image)
            # add the predicted image in the imagePredicted for consensus
            imagePredicted += curr_image

        imagePredicted /= len(arrayTimeIndexImages.keys())

    return imagePredicted, categoricalImage


################################################################################
# Function to predict an image starting from the parametric maps
def generateImageFromParametricMaps(nn, test_df):
    """
    Generate a 2D image from the test_df using the parametric maps

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - checkImageProcessed       : the labeled image (Ground truth img)
    - relativePatientFolderTMP  : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - imagePredicted        : the predicted image
    """

    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    categoricalImage = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, constants.N_CLASSES))
    startX, startY = 0, 0

    while True:
        row_to_analyze = test_df[test_df.x_y == (startX, startY)]

        assert len(row_to_analyze) == 1, "The length of the row to analyze should be 1."

        binary_mask = np.zeros(shape=(constants.getM(), constants.getN()))
        pms = dict()
        for pm_name in constants.getList_PMS():
            filename = row_to_analyze[pm_name].iloc[0]
            pm = cv2.imread(filename, nn.inputImgFlag)
            pms[pm_name] = general_utils.getSlicingWindow(pm, startX, startY, removeColorBar=True)
            # add the mask of the pixels that are > 0 only if it's the MIP image
            if pm_name=="MIP":
                if nn.params["convertImgToGray"]: binary_mask += pms[pm_name] > 0
                else: binary_mask += (cv2.cvtColor(pms[pm_name], cv2.COLOR_RGB2GRAY) > 0)
            pms[pm_name] = np.array(pms[pm_name])
            pms[pm_name] = pms[pm_name].reshape((1,) + pms[pm_name].shape)
            if nn.params["concatenate_input"] and nn.params["inflate_network"]: pms[pm_name] = pms[pm_name].reshape((1,)+pms[pm_name].shape)
            elif nn.params["concatenate_input"]: pms[pm_name] = pms[pm_name].reshape(pms[pm_name].shape + (1,))

        X = []
        if "cbf" in nn.multiInput.keys() and nn.multiInput["cbf"] == 1: X.append(pms["CBF"])
        if "cbv" in nn.multiInput.keys() and nn.multiInput["cbv"] == 1: X.append(pms["CBV"])
        if "ttp" in nn.multiInput.keys() and nn.multiInput["ttp"] == 1: X.append(pms["TTP"])
        if "mtt" in nn.multiInput.keys() and nn.multiInput["mtt"] == 1: X.append(pms["MTT"])
        if "tmax" in nn.multiInput.keys() and nn.multiInput["tmax"] == 1: X.append(pms["TMAX"])
        if "mip" in nn.multiInput.keys() and nn.multiInput["mip"] == 1: X.append(pms["MIP"])
        if "nihss" in nn.multiInput.keys() and nn.multiInput["nihss"] == 1: X.append(np.array([int(row_to_analyze["NIHSS"].iloc[0])]) if row_to_analyze["NIHSS"].iloc[0]!="-" else np.array([0]))
        if "age" in nn.multiInput.keys() and nn.multiInput["age"] == 1: X.append(np.array([int(row_to_analyze["age"].iloc[0])]))
        if "gender" in nn.multiInput.keys() and nn.multiInput["gender"] == 1: X.append(np.array([int(row_to_analyze["gender"].iloc[0])]))

        # slicingWindowPredicted contain only the prediction for the last step
        imagePredicted, categoricalImage = generate2DImage(nn, X, (startX,startY), imagePredicted, categoricalImage, binary_mask)

        # if we reach the end of the image, break the while loop.
        if startX>= constants.IMAGE_WIDTH- constants.getM() and startY>= constants.IMAGE_HEIGHT- constants.getN(): break

        # check for M == WIDTH & N == HEIGHT
        if constants.getM()== constants.IMAGE_WIDTH and constants.getN()== constants.IMAGE_HEIGHT: break

        # going to the next slicingWindow
        if startY<=(constants.IMAGE_HEIGHT - constants.getN()): startY+= constants.getN()
        else:
            if startX < constants.IMAGE_WIDTH:
                startY = 0
                startX += constants.getM()

    return imagePredicted, categoricalImage


################################################################################
# Test the model with the selected patient
def evaluateModel(nn, p_id, isAlreadySaved):
    suffix = general_utils.getSuffix()

    if isAlreadySaved:
        suffix_filename = ".pkl"
        if nn.use_hickle: suffix_filename = ".hkl"
        filename_train = nn.datasetFolder + constants.DATASET_PREFIX + str(p_id) + suffix + suffix_filename

        if not os.path.exists(filename_train): return

        nn.train_df = dataset_utils.readFromPickleOrHickle(filename_train, nn.use_hickle)

        nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, nn.use_sequence, nn.mp_in_nn)
        if not nn.use_sequence: nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, dataset=nn.dataset["test"], modelname=nn.name, flag="test")
        nn.compileModel()  # compile the model and then evaluate it

    sample_weights = nn.getSampleWeights("test")
    if nn.use_sequence:
        multiplier = 16

        nn.test_sequence = sequence_utils.datasetSequence(
            dataframe=nn.train_df,
            indices=nn.dataset["test"]["indices"],
            sample_weights=sample_weights,
            x_label=nn.x_label,
            y_label=nn.y_label,
            multiInput=nn.multiInput,
            params=nn.params,
            batch_size=nn.batch_size,
            flagtype="test",
            back_perc=100,
            loss=nn.loss["name"],
            is4DModel=nn.is4DModel,
            inputImgFlag=nn.inputImgFlag,
            supervised=nn.supervised,
            patientsFolder=nn.patientsFolder
        )

        testing = nn.model.evaluate_generator(
            generator=nn.test_sequence,
            max_queue_size=10*multiplier,
            workers=1*multiplier,
            use_multiprocessing=nn.mp_in_nn
        )

    else:
        testing = nn.model.evaluate(
            x=nn.dataset["test"]["data"],
            y=nn.dataset["test"]["labels"],
            callbacks=nn.callbacks,
            sample_weight=sample_weights,
            verbose=constants.getVerbose(),
            batch_size=nn.batch_size,
            use_multiprocessing=nn.mp_in_nn
        )

    general_utils.printSeparation("-",50)
    if not isAlreadySaved:
        for metric_name in nn.train.history:
            print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.printSeparation("-",50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID()+suffix+".txt", "a+") as text_file:
        if not isAlreadySaved:
            for metric_name in nn.train.history:
                text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing):
            text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
        text_file.write("----------------------------------------------------- \n")
