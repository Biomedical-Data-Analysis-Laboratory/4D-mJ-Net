# Run the testing function, save the images ..
import cv2
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import time
from tensorflow.keras import models
import warnings
import pickle as pkl
import seaborn as sns
import tensorflow.keras.backend as K
from scipy import ndimage
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from Model.constants import *
from Utils import general_utils, dataset_utils, sequence_utils, metrics, model_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Get the labeled image processed (= GT)
def getCheckImageProcessed(nn, p_id, idx):
    checkImageProcessed = np.zeros(shape=(get_img_width(), get_img_weight()))
    # get the label image only if the path is set
    if nn.labeledImagesFolder != "":
        filename = nn.labeledImagesFolder + get_prefix_img() + p_id + os.path.sep + idx + SUFFIX_IMG
        if not os.path.exists(filename):
            print("[WARNING] - {0} does NOT exists, try another...".format(filename))
            filename = nn.labeledImagesFolder + get_prefix_img() + p_id + os.path.sep + idx + ".png"
            if not os.path.exists(filename):
                print("[WARNING] - {0} does NOT exists, try another...".format(filename))
                filename = nn.labeledImagesFolder + get_prefix_img() + p_id + os.path.sep + p_id + idx + SUFFIX_IMG
                assert os.path.exists(filename), "[ERROR] - {0} does NOT exist".format(filename)

        checkImageProcessed = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        if len(checkImageProcessed.shape)==3: checkImageProcessed=cv2.cvtColor(checkImageProcessed, cv2.COLOR_BGR2GRAY)
        assert len(checkImageProcessed.shape)==2, "The GT image shape should be 2."
    return checkImageProcessed


################################################################################
# Predict the model based on the input
def predictFromModel(nn, x_input):
    preds = nn.model.predict(x=x_input, batch_size=1, use_multiprocessing=nn.mp_in_nn)

    return preds


################################################################################
# Save the intermediate layers
def saveIntermediateLayers(model, x, idx, intermediate_activation_path):
    if not os.path.isdir(intermediate_activation_path + idx): os.mkdir(intermediate_activation_path + idx)
    for layer in model.layers:
        if "conv" in layer.name or "leaky" in layer.name or "batch" in layer.name:
            layer_output = model.get_layer(layer.name).output
            intermediate_model = models.Model(inputs=model.input, outputs=layer_output)

            intermediate_prediction = intermediate_model.predict(x, batch_size=1)
            print(layer.name, intermediate_prediction.shape)
            if not os.path.isdir(intermediate_activation_path + idx + os.path.sep + layer.name): os.mkdir(intermediate_activation_path + idx + os.path.sep + layer.name)

            for img_index in range(0, intermediate_prediction.shape[1]):
                plt.figure(figsize=(30, 30))
                xydim = int(np.ceil(np.sqrt(intermediate_prediction.shape[4])))
                for c in range(0, intermediate_prediction.shape[4]):
                    plt.subplot(xydim, xydim, c+1), plt.imshow(intermediate_prediction[0, img_index, :, :, c], cmap='gray'), plt.axis('off')
                plt.savefig(intermediate_activation_path + idx + os.path.sep + layer.name + os.path.sep + str(img_index) + ".png")
                # plt.show()
                    # cv2.imwrite(intermediate_activation_path + layer.name + os.path.sep + str(img_index) + "_" + str(c) + ".png",
                    #             intermediate_prediction[0, img_index, :, :, c])


################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(nn, p_id):
    suffix = general_utils.get_suffix()  # es == "_4_16x16"

    suffix_filename = ".pkl"
    if nn.use_hickle: suffix_filename = ".hkl"
    filename_test = nn.datasetFolder + DATASET_PREFIX + str(p_id) + suffix + suffix_filename

    if not os.path.exists(filename_test): return

    relativePatientFolder = get_prefix_img() + str(p_id) + os.path.sep
    relativePatientFolderHeatMap = relativePatientFolder + "HEATMAP" + os.path.sep
    relativePatientFolderGT = relativePatientFolder + "GT" + os.path.sep
    relativePatientFolderTMP = relativePatientFolder + "TMP" + os.path.sep
    patientFolder = nn.patientsFolder+relativePatientFolder

    filename_saveImageFolder = nn.saveImagesFolder+nn.experimentID+"__"+nn.getNNID()+suffix
    # create the related folders
    general_utils.create_dir(filename_saveImageFolder)
    for subpath in [relativePatientFolder,relativePatientFolderHeatMap,relativePatientFolderGT,relativePatientFolderTMP]:
        general_utils.create_dir(filename_saveImageFolder + os.path.sep + subpath)

    prefix = nn.experimentID + suffix_partial_weights + nn.getNNID() + suffix + os.path.sep
    subpatientFolder = prefix+relativePatientFolder
    patientFolderHeatMap = prefix+relativePatientFolderHeatMap
    patientFolderGT = prefix+relativePatientFolderGT
    patientFolderTMP = prefix+relativePatientFolderTMP

    # for all the slice folders in patientFolder
    for subfolder in glob.glob(patientFolder+"*"+os.path.sep):
        # Predict the images
        if getUSE_PM(): predictImagesFromParametricMaps(nn, subfolder, p_id, subpatientFolder, patientFolderHeatMap, patientFolderGT, patientFolderTMP, filename_test)
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
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(get_img_width(), get_img_weight()))
    categoricalImage = np.zeros(shape=(get_img_width(), get_img_weight(), get_n_classes()))

    idx = general_utils.get_str_from_idx(subfolder.replace(patientFolder, '').replace(os.path.sep, ""))  # image index
    # remove the old logs.
    logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    if os.path.isfile(logsName): os.remove(logsName)

    if is_verbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    checkImageProcessed = getCheckImageProcessed(nn, str(p_id), idx)
    binary_mask = checkImageProcessed != get_pixel_values()[0]

    # Portion for the prediction of the image
    if is_3D() != "":
        assert os.path.exists(filename_test), "[ERROR] - File {} does NOT exist".format(filename_test)

        test_df = dataset_utils.read_pickle_or_hickle(filename_test, nn.use_hickle)
        # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
        test_df = test_df[test_df.data_aug_idx==0]
        print(test_df.shape)
        test_df = test_df[test_df.sliceIndex==idx]
        print(test_df.shape)
        imagePredicted = generateTimeImagesAndConsensus(nn, test_df, relativepatientFolderGT, relativePatientFolderTMP, idx)
    else:  # usual behaviour
        while True:
            test_df = dataset_utils.read_pickle_or_hickle(filename_test, nn.use_hickle)
            test_df = test_df[test_df.x_y == (startingX, startingY)]
            test_df = test_df[test_df.sliceIndex == idx]
            row = test_df[test_df.data_aug_idx == 0]
            # Control that the analyzed row is == 1
            assert len(row) == 1, "The length of the row to analyze should be 1."
            X = model_utils.getCorrectXForInputModel(nn.test_sequence, subfolder, row, batch_idx=0, batch_len=1)

            imagePredicted, categoricalImage = generate2DImage(nn, X, (startingX,startingY), imagePredicted, categoricalImage, binary_mask, idx)

            # if we reach the end of the image, break the while loop.
            if startingX>=get_img_width() -get_m() and startingY>=get_img_weight() -get_n(): break

            # going to the next slicingWindow
            if startingY< get_img_weight() - get_n(): startingY+= get_n()
            else:
                if startingX< get_img_width():
                    startingY=0
                    startingX+= get_m()

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
    if is_ISLES2018(): n_fold = 5

    if len(glob.glob(subfolder+"*"+os.path.sep))>=n_fold:
        pmcheckfold = os.path.sep+"CBF"+os.path.sep
        for idx in glob.glob(subfolder+pmcheckfold+"*"):
            idx = general_utils.get_str_from_idx(idx.replace(subfolder, '').replace(pmcheckfold, ""))  # image index
            if is_ISLES2018(): idx = idx.replace(".tiff", "")
            else: idx = idx.replace(".png","")
            # remove the old logs.
            logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
            if os.path.isfile(logsName): os.remove(logsName)
    
            # if getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))

            checkImageProcessed = getCheckImageProcessed(nn, str(p_id), idx)

            assert os.path.exists(filename_test), "[ERROR] - File {0} does NOT exist".format(filename_test)
    
            # get the pandas dataframe
            test_df = dataset_utils.read_pickle_or_hickle(filename_test, nn.use_hickle)
            # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
            test_df = test_df[test_df.data_aug_idx == 0]
            test_df = test_df[test_df.sliceIndex == idx]
    
            imagePredicted, categoricalImage = generateImageFromParametricMaps(nn, test_df, idx)
    
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
        if is_TO_CATEG() and nn.labeledImagesFolder!= "":
            p_idx, c_idx = 2,3
            if get_n_classes() ==3:p_idx, c_idx = 1, 2
            elif get_n_classes() ==2: c_idx = 1

            checkImageProcessed_rgb = cv2.cvtColor(checkImageProcessed, cv2.COLOR_GRAY2RGB)
            if get_n_classes() >= 3:
                heatmap_img_p = cv2.convertScaleAbs(categoricalImage[:, :, p_idx] * 255)
                heatmap_img_p = cv2.applyColorMap(heatmap_img_p, cv2.COLORMAP_JET)
                blend_p = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_p, 0.5, 0.0)
                cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_penumbra.png", blend_p)
            heatmap_img_c = cv2.convertScaleAbs(categoricalImage[:, :, c_idx] * 255)
            heatmap_img_c = cv2.applyColorMap(heatmap_img_c, cv2.COLORMAP_JET)
            blend_c = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_c, 0.5, 0.0)
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderHeatMap + idx + "_heatmap_core.png", blend_c)

        # Save the ground truth and the contours
        if is_3D() == "" and nn.labeledImagesFolder!= "":
            # save the GT
            cv2.imwrite(nn.saveImagesFolder + relativepatientFolderGT + idx + SUFFIX_IMG, checkImageProcessed)

            imagePredicted = cv2.cvtColor(np.uint8(imagePredicted),cv2.COLOR_GRAY2RGB)  # back to rgb
            if get_n_classes() >= 3:
                _, penumbra_mask = cv2.threshold(checkImageProcessed, 85, get_pixel_values()[-2], cv2.THRESH_BINARY)
                penumbra_cnt, _ = cv2.findContours(penumbra_mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                imagePredicted = cv2.drawContours(imagePredicted, penumbra_cnt, -1, (255,0,0), 2)
            _, core_mask = cv2.threshold(checkImageProcessed, get_pixel_values()[-2], get_pixel_values()[-1], cv2.THRESH_BINARY)
            core_cnt, _ = cv2.findContours(core_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imagePredicted = cv2.drawContours(imagePredicted, core_cnt, -1, (0,0,255), 2)
            # save the GT image with predicted contours
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, penumbra_area, 0.5, 0.0)
            # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, core_area, 0.5, 0.0)
            cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+idx+".png",imagePredicted)


################################################################################
# Helpful function that return the 2D image from the pixel and the starting coordinates
def generate2DImage(nn, pixels, startingXY, imgPredicted, categoricalImage, binary_mask, idx):
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
    if nn.save_images and is_TO_CATEG(): categoricalImage[x:x + get_m(), y:y + get_n()]=swp_orig

    # convert the categorical into a single array for removing some uncertain predictions
    if is_TO_CATEG(): slicingWindowPredicted = K.eval((K.argmax(swp_orig) * 255) / (get_n_classes() - 1))
    else: slicingWindowPredicted = swp_orig * 255
    # save the predicted images
    if nn.save_images:
        if not has_limited_columns() and get_n_classes() >2:
            # Remove the parts already classified by the model
            binary_mask = np.array(binary_mask, dtype=np.float)
            # force all the predictions to be inside the binary mask defined by the GT
            slicingWindowPredicted *= binary_mask[x:x + get_m(), y:y + get_n()]

            overlapping_pred = np.array(slicingWindowPredicted>0,dtype=np.float)
            overlapping_pred *= 85.
            binary_mask *= 85.  # multiply the binary mask for the brain pixel value
            # add the brain to the prediction window
            slicingWindowPredicted += (binary_mask[x:x + get_m(), y:y + get_n()] - overlapping_pred)
        imgPredicted[x:x + get_m(), y:y + get_n()]=slicingWindowPredicted

    if nn.save_activation_filter:
        if idx == "03" or idx == "04": saveIntermediateLayers(nn.model, pixels, idx, intermediate_activation_path=nn.intermediateActivationFolder)

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

    imagePredicted = np.zeros(shape=(get_img_width(), get_img_weight()))
    categoricalImage = np.zeros(shape=(get_img_width(), get_img_weight(), get_n_classes()))
    checkImageProcessed = np.zeros(shape=(get_img_width(), get_img_weight()))
    arrayTimeIndexImages = dict()

    for test_row in test_df.itertuples():  # for every rows of the same image
        if str(test_row.timeIndex) not in arrayTimeIndexImages.keys(): arrayTimeIndexImages[str(test_row.timeIndex)] = np.zeros(shape=(
            get_img_width(), get_img_weight()), dtype=np.uint8)
        if is_3D() == "": test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2], 1)
        else: test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2])
        arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage = generate2DImage(nn, test_row.pixels, test_row.x_y, arrayTimeIndexImages[str(test_row.timeIndex)], categoricalImage, checkImageProcessed, idx)

    if nn.save_images:              # remove one class from the ground truth
        if get_n_classes() ==3: checkImageProcessed[checkImageProcessed == 85] = get_pixel_values()[0]
        cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP +"orig_" + idx + SUFFIX_IMG, checkImageProcessed)

        for tidx in arrayTimeIndexImages.keys():
            curr_image = arrayTimeIndexImages[tidx]
            # save the images
            cv2.imwrite(nn.saveImagesFolder + relativePatientFolderTMP + idx +"_" + general_utils.get_str_from_idx(
                tidx) + SUFFIX_IMG, curr_image)
            # add the predicted image in the imagePredicted for consensus
            imagePredicted += curr_image

        imagePredicted /= len(arrayTimeIndexImages.keys())

    return imagePredicted, categoricalImage


################################################################################
# Function to predict an image starting from the parametric maps
def generateImageFromParametricMaps(nn, test_df, idx):
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

    imagePredicted = np.zeros(shape=(get_img_width(), get_img_weight()))
    categoricalImage = np.zeros(shape=(get_img_width(), get_img_weight(), get_n_classes()))
    startX, startY = 0, 0
    constants ={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                "weights": get_weights(), "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(), "USE_PM":getUSE_PM(),
                "LIST_PMS":get_list_PMS(), "IMAGE_HEIGHT":get_img_weight(), "IMAGE_WIDTH": get_img_width()}

    while True:
        row_to_analyze = test_df[test_df.x_y == (startX, startY)]

        assert len(row_to_analyze) == 1, "The length of the row to analyze should be 1."

        binary_mask = np.zeros(shape=(get_m(), get_n()))
        pms = dict()
        for pm_name in get_list_PMS():
            filename = row_to_analyze[pm_name].iloc[0]
            pm = cv2.imread(filename, nn.inputImgFlag)
            pms[pm_name] = general_utils.getSlicingWindow(pm, startX, startY, constants, removeColorBar=True)
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
        imagePredicted, categoricalImage = generate2DImage(nn, X, (startX,startY), imagePredicted, categoricalImage, binary_mask, idx)

        # if we reach the end of the image, break the while loop.
        if startX>= get_img_width() - get_m() and startY>= get_img_weight() - get_n(): break

        # check for M == WIDTH & N == HEIGHT
        if get_m() == get_img_width() and get_n() == get_img_weight(): break

        # going to the next slicingWindow
        if startY<=(get_img_weight() - get_n()): startY+= get_n()
        else:
            if startX < get_img_width():
                startY = 0
                startX += get_m()

    return imagePredicted, categoricalImage


################################################################################
# Test the model with the selected patient
def evaluateModel(nn, p_id, isAlreadySaved):
    suffix = general_utils.get_suffix()

    if isAlreadySaved:
        suffix_filename = ".pkl"
        if nn.use_hickle: suffix_filename = ".hkl"
        filename_train = nn.datasetFolder + DATASET_PREFIX + str(p_id) + suffix + suffix_filename

        if not os.path.exists(filename_train): return

        nn.train_df = dataset_utils.read_pickle_or_hickle(filename_train, nn.use_hickle)

        nn.dataset = dataset_utils.get_test_ds(nn.dataset, nn.train_df, p_id, nn.use_sequence, nn.mp_in_nn)
        if not nn.use_sequence: nn.dataset["test"]["labels"] = dataset_utils.get_labels_from_idx(train_df=nn.train_df,
                                                                                                 dataset=nn.dataset[
                                                                                                     "test"],
                                                                                                 modelname=nn.name,
                                                                                                 flag="test")
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
            name=nn.name,
            is3dot5DModel=nn.is3dot5DModel,
            is4DModel=nn.is4DModel,
            inputImgFlag=nn.inputImgFlag,
            supervised=nn.supervised,
            patientsFolder=nn.patientsFolder,
            labeledImagesFolder=nn.labeledImagesFolder,
            constants={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                       "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                       "weights": get_weights(), "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(), "USE_PM":getUSE_PM(),
                       "LIST_PMS":get_list_PMS(), "IMAGE_HEIGHT":get_img_weight(), "IMAGE_WIDTH": get_img_width()}
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
            verbose=is_verbose(),
            batch_size=nn.batch_size,
            use_multiprocessing=nn.mp_in_nn
        )

    general_utils.print_sep("-", 50)
    if not isAlreadySaved:
        for metric_name in nn.train.history:
            print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.print_sep("-", 50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID()+suffix+".txt", "a+") as text_file:
        if not isAlreadySaved:
            for metric_name in nn.train.history:
                text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing):
            text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
        text_file.write("----------------------------------------------------- \n")
