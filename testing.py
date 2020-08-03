# Run the testing function, save the images ..
from Utils import general_utils, dataset_utils, metrics
import constants, training

import os, time, cv2, glob
import multiprocessing
import numpy as np
import tensorflow.keras.backend as K

def predictFromModel(nn, input):
    return nn.model.predict(
            x=input,
            batch_size=nn.batch_size,
            steps=nn.test_steps,
            use_multiprocessing=nn.mp
    )

################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(nn, p_id):
    start = time.time()
    stats = {}
    suffix = general_utils.getSuffix() ## es == "_4_16x16"

    suffix_filename = ".pkl"
    if nn.use_hickle: suffix_filename = ".hkl"
    filename_test = nn.datasetFolder+constants.DATASET_PREFIX+str(p_id)+suffix+suffix_filename

    relativePatientFolder = constants.getPrefixImages()+p_id+"/"
    relativePatientFolderHeatMap = relativePatientFolder + "HEATMAP/"
    relativePatientFolderTMP = relativePatientFolder + "TMP/"
    patientFolder = nn.patientsFolder+relativePatientFolder

    # create the related folders
    general_utils.createDir(nn.saveImagesFolder+nn.getNNID(p_id)+suffix)
    general_utils.createDir(nn.saveImagesFolder+nn.getNNID(p_id)+suffix+"/"+relativePatientFolder)
    general_utils.createDir(nn.saveImagesFolder+nn.getNNID(p_id)+suffix+"/"+relativePatientFolderHeatMap)
    general_utils.createDir(nn.saveImagesFolder+nn.getNNID(p_id)+suffix+"/"+relativePatientFolderTMP)

    if constants.getVerbose(): general_utils.printSeparation("-", 100)

    # for all the slice folders in patientFolder
    for subfolder in glob.glob(patientFolder+"*/"):
        # try:
        subpatientFolder = nn.getNNID(p_id)+suffix+"/"+relativePatientFolder
        patientFolderHeatMap = nn.getNNID(p_id)+suffix+"/"+relativePatientFolderHeatMap
        patientFolderTMP = nn.getNNID(p_id)+suffix+"/"+relativePatientFolderTMP

        tmpStats = predictImage(nn, subfolder, p_id, patientFolder, subpatientFolder, patientFolderHeatMap, patientFolderTMP, filename_test)

        if nn.save_statistics:
            for func in nn.statistics:
                if func.__name__ not in stats.keys(): stats[func.__name__] = {}
                for classToEval in nn.classes_to_evaluate:
                    if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = {}
                    for idxE, _ in enumerate(nn.epsiloList):
                        if idxE not in stats[func.__name__][classToEval].keys(): stats[func.__name__][classToEval][idxE] = []
                        stats[func.__name__][classToEval][idxE].append(tmpStats[func.__name__][classToEval][idxE])
        # except Exception as e:
        #     print("[ERROR] - ", e)
        #     continue

    end = time.time()
    if constants.getVerbose():
        general_utils.printSeparation("-", 100)
        print("[INFO] - Total time: {0}s for patient {1}.".format(round(end-start, 3), p_id))
        general_utils.printSeparation("-", 100)

    return stats

################################################################################
def predictImage(nn, subfolder, p_id, patientFolder, relativePatientFolder, relativePatientFolderHeatMap, relativePatientFolderTMP, filename_test):
    """
    Generate a SINGLE image for the patient and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - relativePatientFolder         : relative name of the patient folder
    - relativePatientFolderHeatMap  : relative name of the patient heatmap folder
    - relativePatientFolderTMP      : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe

    Return:
    - stats                         : statistics dictionary
    """

    start = time.time()
    stats = {}
    imagesDict = {} # faster access to the images
    YTRUEToEvaluate, YPREDToEvaluate = [], []
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))

    idx = general_utils.getStringFromIndex(subfolder.replace(patientFolder, '').replace("/", "")) # image index

    # remove the old logs.
    logsName = nn.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    if os.path.isfile(logsName): os.remove(logsName)

    if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    s1 = time.time()

    # get the label image only if the path is set
    if nn.labeledImagesFolder!="":
        filename = nn.labeledImagesFolder+constants.PREFIX_IMAGES+p_id+"/"+idx+constants.SUFFIX_IMG
        if not os.path.exists(filename):
            print("[WARNING] - {0} does NOT exists, try another...".format(filename))
            filename = nn.labeledImagesFolder+constants.PREFIX_IMAGES+p_id+"/"+p_id+idx+constants.SUFFIX_IMG
            if not os.path.exists(filename):
                raise Exception("[ERROR] - {0} does NOT exist".format(filename))

        checkImageProcessed = cv2.imread(filename, 0)

    # get the images in a dictionary
    for imagename in np.sort(glob.glob(subfolder+"*"+constants.SUFFIX_IMG)): # sort the images !
        filename = imagename.replace(subfolder, '')
        if not nn.supervised or nn.patientsFolder!="OLDPREPROC_PATIENTS/": imagesDict[filename] = cv2.imread(imagename, 0)
        else: # don't take the first image (the manually annotated one)
            if filename != "01"+constants.SUFFIX_IMG: imagesDict[filename] = cv2.imread(imagename, 0)

    # Portion for the prediction of the image
    if constants.get3DFlag()!="":
        if not os.path.exists(filename_test):
            if constants.getVerbose(): print("[WARNING] - File {} does NOT exist".format(filename_test))
            return stats

        test_df = dataset_utils.readFromPickleOrHickle(filename_test, nn.use_hickle)
        test_df = test_df[test_df.data_aug_idx==0] # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
        test_df = test_df[test_df.timeIndex==idx]
        imagePredicted, YTRUEToEvaluate, YPREDToEvaluate = generateTimeImagesAndConsensus(nn, test_df, YTRUEToEvaluate, YPREDToEvaluate, relativePatientFolderTMP, idx)
    else:
        while True:
            pixels_shape = (constants.getM(),constants.getN(),constants.NUMBER_OF_IMAGE_PER_SECTION)
            if constants.ORIGINAL_SHAPE: pixels_shape = (constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN())

            pixels = np.zeros(shape=pixels_shape)
            count = 0

            # for each image get the array for prediction
            for imagename in np.sort(glob.glob(subfolder+"*"+constants.SUFFIX_IMG)):
                filename = imagename.replace(subfolder, '')
                if not nn.supervised or nn.patientsFolder!="OLDPREPROC_PATIENTS/":
                    image = imagesDict[filename]
                    if constants.ORIGINAL_SHAPE: pixels[count,:,:] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                    else: pixels[:,:,count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                    count+=1
                else:
                    if filename != "01"+constants.SUFFIX_IMG:
                        image = imagesDict[filename]
                        if constants.ORIGINAL_SHAPE: pixels[count,:,:] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                        else: pixels[:,:,count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                        count+=1

            tileImageProcessed = general_utils.getSlicingWindow(checkImageProcessed, startingX, startingY, constants.getM(), constants.getN())
            imagePredicted, YTRUEToEvaluate, YPREDToEvaluate = generate2DImage(nn, pixels, (startingX,startingY), imagePredicted, tileImageProcessed, YTRUEToEvaluate, YPREDToEvaluate)

            # if we reach the end of the image, break the while loop.
            if startingX>=constants.IMAGE_WIDTH-constants.getM() and startingY>=constants.IMAGE_HEIGHT-constants.getN(): break

            # going to the next slicingWindow
            if startingY<constants.IMAGE_HEIGHT-constants.getN(): startingY+=constants.getN()
            else:
                if startingX<constants.IMAGE_WIDTH:
                    startingY=0
                    startingX+=constants.getM()

    s2 = time.time()
    if constants.getVerbose(): print("image time: {}".format(round(s2-s1, 3)))
    if nn.save_images:
        s1 = time.time()
        # rotate the predictions foor the ISLES2018 dataset
        # if "ISLES2018" in nn.datasetFolder: imagePredicted = np.rot90(imagePredicted,-1)
        # save the image predicted in the specific folder
        cv2.imwrite(nn.saveImagesFolder+relativePatientFolder+idx+constants.SUFFIX_IMG, imagePredicted)
        # create and save the HEATMAP
        heatmap_img = cv2.normalize(imagePredicted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_img = cv2.applyColorMap(~heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite(nn.saveImagesFolder+relativePatientFolderHeatMap+idx+"_heatmap"+constants.SUFFIX_IMG, heatmap_img)

        if constants.get3DFlag()=="": cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+idx+constants.SUFFIX_IMG, checkImageProcessed)

        s2 = time.time()
        if constants.getVerbose(): print("save time: {}".format(round(s2-s1, 3)))

    if nn.save_statistics:
        s1 = time.time()
        tn, fn, fp, tp = {}, {}, {}, {}
        for classToEval in nn.classes_to_evaluate:
            if classToEval=="penumbra": label=2
            elif classToEval=="core":
                label=3
                if constants.N_CLASSES==2: label=1 # binary classification
            elif classToEval=="penumbracore": label=4

            if classToEval not in tn.keys(): tn[classToEval] = {}
            if classToEval not in fn.keys(): fn[classToEval] = {}
            if classToEval not in fp.keys(): fp[classToEval] = {}
            if classToEval not in tp.keys(): tp[classToEval] = {}

            for idxE, percEps in enumerate(nn.epsiloList): tn[classToEval][idxE], fn[classToEval][idxE], fp[classToEval][idxE], tp[classToEval][idxE] = metrics.mappingPrediction(YTRUEToEvaluate, YPREDToEvaluate, nn.use_background_in_statistics, nn.epsilons, percEps, label)

        for func in nn.statistics:
            if func.__name__ not in stats.keys(): stats[func.__name__] = {}
            for classToEval in nn.classes_to_evaluate:
                if classToEval=="penumbra": label=2
                elif classToEval=="core":
                    label=3
                    if constants.N_CLASSES==2: label=1 # binary classification
                elif classToEval=="penumbracore": label=4

                if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = {}

                for idxE, _ in enumerate(nn.epsiloList): stats[func.__name__][classToEval][idxE] = (tn[classToEval][idxE], fn[classToEval][idxE], fp[classToEval][idxE], tp[classToEval][idxE])

        s2 = time.time()
        if constants.getVerbose(): print("stats time: {}".format(round(s2-s1, 3)))

    end = time.time()
    if constants.getVerbose():
        print("[INFO] - Time: {0}s for image {1}.".format(round(end-start, 3), idx))
        general_utils.printSeparation("-", 100)

    return stats

################################################################################
def generate2DImage(nn, pixels, startingXY, imagePredicted, checkImageProcessed, YTRUEToEvaluate, YPREDToEvaluate):
    """
    Generate a 2D image from the test_df

    Input parameters:
    - nn                    : NeuralNetwork class
    - pixels                : pixel in a numpy array
    - startingXY            : (x,y) coordinates
    - imagePredicted        : the predicted image
    - checkImageProcessed   : the ground truth image
    - YTRUEToEvaluate       : list for stats corresponding to the ground truth image
    - YPREDToEvaluate       : list for stats corresponding to the predicted image

    Return:
    - imagePredicted        : the predicted image
    - YTRUEToEvaluate       : list for stats corresponding to the ground truth image
    - YPREDToEvaluate       : list for stats corresponding to the predicted image
    """
    startingX, startingY = startingXY
    if constants.get3DFlag()=="": pixels = pixels.reshape(1, pixels.shape[0], pixels.shape[1], pixels.shape[2], 1)
    else: pixels = pixels.reshape(1, pixels.shape[0], pixels.shape[1], pixels.shape[2])

    # slicingWindowPredicted contain only the prediction for the last step
    slicingWindowPredicted = predictFromModel(nn, pixels)[nn.test_steps-1]


    if nn.to_categ: slicingWindowPredicted = K.eval((K.argmax(slicingWindowPredicted)*255)/len(constants.LABELS))
    else: slicingWindowPredicted *= 255

    if nn.save_statistics and nn.labeledImagesFolder!="": # for statistics purposes
        YPREDToEvaluate.extend(slicingWindowPredicted)
        YTRUEToEvaluate.extend(checkImageProcessed)

    # save the predicted images
    if nn.save_images: imagePredicted[startingX:startingX+constants.getM(), startingY:startingY+constants.getN()] = slicingWindowPredicted

    return (imagePredicted, YTRUEToEvaluate, YPREDToEvaluate)

################################################################################
def generateTimeImagesAndConsensus(nn, test_df, YTRUEToEvaluate, YPREDToEvaluate, relativePatientFolderTMP, idx):
    """
    Generate the image from the 3D sequence of time index (create also these images) with a consensus

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - YTRUEToEvaluate           : list for stats corresponding to the ground truth image
    - YPREDToEvaluate           : list for stats corresponding to the predicted image
    - relativePatientFolderTMP  : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - imagePredicted        : the predicted image
    - YTRUEToEvaluate       : list for stats corresponding to the ground truth image
    - YPREDToEvaluate       : list for stats corresponding to the predicted image
    """

    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))
    arrayTimeIndexImages = dict()

    for test_row in test_df.itertuples(): # for every rows of the same image
        if str(test_row.timeIndex) not in arrayTimeIndexImages.keys(): arrayTimeIndexImages[str(test_row.timeIndex)] = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)
        arrayTimeIndexImages[str(test_row.timeIndex)], checkImageProcessed, YTRUEToEvaluate, YPREDToEvaluate = generate2DImage(nn, test_row.pixels, test_row.x_y, arrayTimeIndexImages[str(test_row.timeIndex)], checkImageProcessed, YTRUEToEvaluate, YPREDToEvaluate)

    if nn.save_images:
        if constants.N_CLASSES==3:
            checkImageProcessed[checkImageProcessed==255] = constants.PIXELVALUES[0] # remove one class from the ground truth
            checkImageProcessed[checkImageProcessed==150] == constants.PIXELVALUES[2] # change the class for core
        cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+"orig_"+idx+constants.SUFFIX_IMG, checkImageProcessed)

        for tidx in arrayTimeIndexImages.keys():
            curr_image = arrayTimeIndexImages[tidx]
            # save the images
            cv2.imwrite(nn.saveImagesFolder+relativePatientFolderTMP+idx+"_"+general_utils.getStringFromIndex(tidx)+constants.SUFFIX_IMG, curr_image)
            # add the predicted image in the imagePredicted for consensus
            imagePredicted += curr_image

        imagePredicted /= len(arrayTimeIndexImages.keys())

    return (imagePredicted, YTRUEToEvaluate, YPREDToEvaluate)

################################################################################
# Test the model with the selected patient
def evaluateModel(nn, p_id, isAlreadySaved):
    suffix = general_utils.getSuffix()

    if isAlreadySaved:
        suffix_filename = ".pkl"
        if nn.use_hickle: suffix_filename = ".hkl"
        filename_train = nn.datasetFolder+constants.DATASET_PREFIX+str(p_id)+suffix+suffix_filename

        if not os.path.exists(filename_train): return

        nn.train_df = dataset_utils.readFromPickleOrHickle(filename_train, nn.use_hickle)

        nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)
        nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, dataset=nn.dataset["test"], modelname=nn.name, to_categ=nn.to_categ, flag="test")
        nn.compileModel() # compile the model and then evaluate it

    sample_weights = nn.getSampleWeights("test")

    testing = nn.model.evaluate(
        x=nn.dataset["test"]["data"],
        y=nn.dataset["test"]["labels"],
        callbacks=nn.callbacks,
        sample_weight=sample_weights,
        verbose=constants.getVerbose(),
        batch_size=nn.batch_size,
        use_multiprocessing=nn.mp
    )

    if not nn.train_on_batch:
        general_utils.printSeparation("-",50)
        if not isAlreadySaved:
            for metric_name in nn.train.history:
                print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing):
            print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
        general_utils.printSeparation("-",50)

        with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID(p_id)+suffix+".txt", "a+") as text_file:
            if not isAlreadySaved:
                for metric_name in nn.train.history:
                    text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
            for index, val in enumerate(testing):
                text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
            text_file.write("----------------------------------------------------- \n")
