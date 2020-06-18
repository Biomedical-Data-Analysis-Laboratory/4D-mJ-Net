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
def predictAndSaveImages(that, p_id):
    start = time.time()
    stats = {}

    relativePatientFolder = constants.PREFIX_IMAGES+p_id+"/"
    relativePatientFolderHeatMap = relativePatientFolder + "HEATMAP/"
    relativePatientFolderTMP = relativePatientFolder + "TMP/"
    patientFolder = that.patientsFolder+relativePatientFolder
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix())
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolder)
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolderHeatMap)
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolderTMP)

    if constants.getVerbose():
        for w in range(2): general_utils.printSeparation("-", 100)

    for subfolder in glob.glob(patientFolder+"*/"):
        try:
            subpatientFolder = that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolder
            patientFolderHeatMap = that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolderHeatMap
            patientFolderTMP = that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolderTMP

            tmpStats = predictImage(that, subfolder, p_id, patientFolder, subpatientFolder, patientFolderHeatMap, patientFolderTMP)

            if that.save_statistics:
                for func in that.statistics:
                    if func.__name__ not in stats.keys(): stats[func.__name__] = {}
                    for classToEval in that.classes_to_evaluate:
                        if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = {}
                        for idxE, _ in enumerate(that.epsiloList):
                            if idxE not in stats[func.__name__][classToEval].keys(): stats[func.__name__][classToEval][idxE] = []
                            stats[func.__name__][classToEval][idxE].append(tmpStats[func.__name__][classToEval][idxE])
        except Exception as e:
            print("[ERROR] - ", e)
            continue

    end = time.time()
    if constants.getVerbose():
        general_utils.printSeparation("-", 100)
        print("[INFO] - Total time: {0}s for patient {1}.".format(round(end-start, 3), p_id))
        general_utils.printSeparation("-", 100)

    return stats

################################################################################
# Generate a SINGLE image for the patient and save it
def predictImage(that, subfolder, p_id, patientFolder, relativePatientFolder, relativePatientFolderHeatMap, relativePatientFolderTMP):
    start = time.time()
    stats = {}
    YTRUEToEvaluate, YPREDToEvaluate = [], []
    imagesDict = {} # faster access to the images
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)
    checkImageProcessed = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)
    # imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, 3), dtype=np.uint8)
    suffix = general_utils.getSuffix() ## es == "_4_16x16"

    idx = general_utils.getStringPatientIndex(subfolder.replace(patientFolder, '').replace("/", "")) # image index

    logsName = that.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    # remove the old logs.
    if os.path.isfile(logsName): os.remove(logsName)

    if constants.getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    s1 = time.time()
    filename_test = that.datasetFolder+constants.DATASET_PREFIX+str(p_id)+suffix+".pkl"
    test_df = dataset_utils.loadSingleTrainingData(that.da, filename_test, p_id)
    test_df = test_df[test_df.data_aug_idx==0] # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
    test_df = test_df[test_df.timeIndex==idx]

    if constants.getVerbose(): print("Rows in dataframe to analyze: {}".format(len(test_df.index)))
    for i_t, test_row in test_df.iterrows(): # for every rows of the same image
        startingX, startingY = test_row.x_y
        pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2], 1)

        # get the label image only if the path is set
        if that.labeledImagesFolder!="":
            y_true = test_row.ground_truth
            # cv2.imwrite(that.saveImagesFolder+relativePatientFolderTMP+idx+"y_true.png", y_true)
        # for q in range(test_row.pixels.shape[0]):  cv2.imwrite(that.saveImagesFolder+relativePatientFolderTMP+idx+"_"+str(q)+"real.png", test_row.pixels[q,:,:])

        ### MODEL PREDICT (image & statistics)

        # slicingWindowPredicted contain only the prediction for the last step
        slicingWindowPredicted = predictFromModel(that, pixels)[that.test_steps-1]
        multiplier = 255

        if that.to_categ:
            slicingWindowPredicted = K.eval((K.argmax(slicingWindowPredicted)*256)/len(constants.LABELS))
            multiplier = 1

        if that.save_statistics and that.labeledImagesFolder!="":
            YTRUEToEvaluate.extend(y_true)
            YPREDToEvaluate.extend(slicingWindowPredicted*multiplier)

        if that.save_images:
            threeDimensionSlicingWindow = np.zeros(shape=(slicingWindowPredicted.shape[0],slicingWindowPredicted.shape[1]), dtype=np.uint8)
            # threeDimensionSlicingWindow = np.zeros(shape=(slicingWindowPredicted.shape[0],slicingWindowPredicted.shape[1], 3), dtype=np.uint8)

            if constants.N_CLASSES == 4:
                thresBack = constants.PIXELVALUES[0]
                thresBrain = constants.PIXELVALUES[1]
                thresPenumbra = constants.PIXELVALUES[2]
                thresCore = constants.PIXELVALUES[3]
                eps1, eps2, eps3 = that.epsilons[0]
            else:
                thresBack = constants.PIXELVALUES[0]
                thresCore = constants.PIXELVALUES[1]
                eps1 = that.epsilons[0]

            # print(startingX,startingX+constants.getM(),startingY,startingY+constants.getN())
            # print(slicingWindowPredicted.shape)
            # print(slicingWindowPredicted)

            imagePredicted[startingX:startingX+constants.getM(), startingY:startingY+constants.getN()] = slicingWindowPredicted*multiplier
            checkImageProcessed[startingX:startingX+constants.getM(), startingY:startingY+constants.getN()] = 255
            # for r, _ in enumerate(slicingWindowPredicted):
            #     threeDimensionSlicingWindow[r] = slicingWindowPredicted[r]*multiplier

                # for c, pixel in enumerate(slicingWindowPredicted[r]):
                #     threeDimensionSlicingWindow[r][c] = (pixel*multiplier,)*3
            # Create the image
            # imagePredicted[startingX:startingX+constants.getM(), startingY:startingY+constants.getN()] = threeDimensionSlicingWindow

    s2 = time.time()
    if constants.getVerbose(): print("image time: {}".format(round(s2-s1, 3)))
    if that.save_images:
        s1 = time.time()
        # rotate the predictions foor the ISLES2018 dataset
        # if "ISLES2018" in that.datasetFolder: imagePredicted = np.rot90(imagePredicted,-1)
        # save the image predicted in the specific folder
        cv2.imwrite(that.saveImagesFolder+relativePatientFolder+idx+".png", imagePredicted)
        # create and save the HEATMAP
        heatmap_img = cv2.applyColorMap(~imagePredicted, cv2.COLORMAP_JET)
        cv2.imwrite(that.saveImagesFolder+relativePatientFolderHeatMap+idx+"_heatmap.png", heatmap_img)

        cv2.imwrite(that.saveImagesFolder+relativePatientFolderTMP+idx+".png", checkImageProcessed)

        s2 = time.time()
        if constants.getVerbose(): print("save time: {}".format(round(s2-s1, 3)))

    if that.save_statistics:
        s1 = time.time()
        tn, fn, fp, tp = {}, {}, {}, {}
        for classToEval in that.classes_to_evaluate:
            if classToEval=="penumbra": label=2
            elif classToEval=="core":
                label=3
                if constants.N_CLASSES==2: label=1 # binary classification
            elif classToEval=="penumbracore": label=4

            if classToEval not in tn.keys(): tn[classToEval] = {}
            if classToEval not in fn.keys(): fn[classToEval] = {}
            if classToEval not in fp.keys(): fp[classToEval] = {}
            if classToEval not in tp.keys(): tp[classToEval] = {}

            if that.epsiloList[0]!=None:
                for  idxE,percEps in enumerate(that.epsiloList):
                    # loop over the various epsilon
                    tn[classToEval][idxE], fn[classToEval][idxE], fp[classToEval][idxE], tp[classToEval][idxE] = metrics.mappingPrediction(YTRUEToEvaluate, YPREDToEvaluate, that.use_background_in_statistics, that.epsilons, percEps, label)

        for func in that.statistics:
            if func.__name__ not in stats.keys(): stats[func.__name__] = {}
            for classToEval in that.classes_to_evaluate:
                if classToEval=="penumbra": label=2
                elif classToEval=="core":
                    label=3
                    if constants.N_CLASSES==2: label=1 # binary classification
                elif classToEval=="penumbracore": label=4

                if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = {}

                if that.epsiloList[0]!=None:
                    for  idxE, _ in enumerate(that.epsiloList):
                        stats[func.__name__][classToEval][idxE] = (tn[classToEval][idxE], fn[classToEval][idxE], fp[classToEval][idxE], tp[classToEval][idxE])
        s2 = time.time()
        if constants.getVerbose(): print("stats time: {}".format(round(s2-s1, 3)))
    end = time.time()
    if constants.getVerbose():
        print("[INFO] - Time: {0}s for image {1}.".format(round(end-start, 3), idx))
        general_utils.printSeparation("-", 100)

    return stats

################################################################################
# Test the model with the selected patient
def evaluateModel(nn, p_id, isAlreadySaved):
    suffix = general_utils.getSuffix()
    if isAlreadySaved:
        filename_train = nn.datasetFolder+constants.DATASET_PREFIX+str(p_id)+suffix+".pkl"
        nn.train_df = dataset_utils.readFromHickle(filename_train)
        nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)
        nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, indices=nn.dataset["test"]["indices"], to_categ=nn.to_categ, flag="train")
        nn.compileModel() # compile the model and then evaluate

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
