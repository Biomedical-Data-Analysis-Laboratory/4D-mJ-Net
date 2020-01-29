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
            steps=nn.test_steps,
            #callbacks=nn.callbacks,
            use_multiprocessing=nn.mp
    )

################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(that, p_id):
    start = time.time()
    stats = {}

    relativePatientFolder = constants.PREFIX_IMAGES+p_id+"/"
    patientFolder = that.patientsFolder+relativePatientFolder
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix())
    general_utils.createDir(that.saveImagesFolder+that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolder)

    if constants.getVerbose(): general_utils.printSeparation("-", 100)

    # TODO: reduce the prediction time to <60s!
    # TODO: find a way to use multiprocessing the generation of the images!!!!
    # if that.mp:
    #     cpu_count = multiprocessing.cpu_count()
    #     input = []
    #     for subfolder in glob.glob(patientFolder+"*/"):
    #         input.append((that, subfolder, p_id, patientFolder, relativePatientFolder))
    #     with multiprocessing.Pool(processes=cpu_count) as pool: # auto closing workers
    #         pool.starmap(predictImage, input)
    # else:


    for subfolder in glob.glob(patientFolder+"*/"):
        tmpStats = predictImage(that, subfolder, p_id, patientFolder, that.getNNID(p_id)+general_utils.getSuffix()+"/"+relativePatientFolder)
        for func in that.statistics:
            if func.__name__ not in stats.keys(): stats[func.__name__] = {}
            for classToEval in that.classes_to_evaluate:
                if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = []
                # meanV = np.mean(tmpStats[func.__name__][classToEval])
                # print("TEST MEAN %s %s: %.2f%% " % (func.__name__, classToEval, round(meanV,6)*100))
                # stats[func.__name__][classToEval].append(meanV)
                stats[func.__name__][classToEval].append(tmpStats[func.__name__][classToEval])
            # general_utils.printSeparation("+",20)

    end = time.time()
    print("Total time: {0}s for patient {1}.".format(round(end-start, 3), p_id))

    if constants.getVerbose(): general_utils.printSeparation("-", 100)

    return stats

################################################################################
# Generate a SINGLE image for the patient and save it
def predictImage(that, subfolder, p_id, patientFolder, relativePatientFolder):
    start = time.time()
    stats = {}
    YTRUEToEvaluate, YPREDToEvaluate = [], []

    idx = general_utils.getStringPatientIndex(subfolder.replace(patientFolder, '').replace("/", "")) # image index

    logsName = that.saveImagesFolder+relativePatientFolder+idx+"_logs.txt"
    # remove the old logs.
    if os.path.isfile(logsName): os.remove(logsName)

    if constants.getVerbose():
        print("Analyzing Patient {0}, image {1}...".format(p_id, idx))

    labeled_image = cv2.imread(that.labeledImagesFolder+"Patient"+p_id+"/"+p_id+idx+".png", 0)
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, 3), dtype=np.uint8)

    imagesDict = {} # faster access to the images
    for imagename in np.sort(glob.glob(subfolder+"*.png")): # sort the images !
        filename = imagename.replace(subfolder, '')
        if not that.supervised or that.patientsFolder!="OLDPREPROC_PATIENTS/":
            image = cv2.imread(imagename, 0)
            imagesDict[filename] = image
        else:
            if filename != "01.png": # don't take the first image (the manually annotated one)
                image = cv2.imread(imagename, 0)
                imagesDict[filename] = image

    # Generate the predicted image
    s1 = time.time()
    while True:
        pixels = np.zeros(shape=(constants.NUMBER_OF_IMAGE_PER_SECTION,constants.getM(),constants.getN()))
        count = 0
        row, column = 0, 0

        # for each image
        for imagename in np.sort(glob.glob(subfolder+"*.png")):
            filename = imagename.replace(subfolder, '')
            if not that.supervised or that.patientsFolder!="OLDPREPROC_PATIENTS/":
                image = imagesDict[filename]
                pixels[count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                count+=1
            else:
                if filename != "01.png":
                    image = imagesDict[filename]
                    pixels[count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.getM(), constants.getN())
                    count+=1

        pixels = pixels.reshape(1, pixels.shape[0], pixels.shape[1], pixels.shape[2], 1)

        ### MODEL PREDICT (image & statistics)
        y_true = general_utils.getSlicingWindow(labeled_image, startingX, startingY, constants.getM(), constants.getN())
        # slicingWindowPredicted contain only the prediction for the last step
        slicingWindowPredicted = predictFromModel(that, pixels)[that.test_steps-1]

        if that.save_statistics:
            YTRUEToEvaluate.extend(y_true)
            YPREDToEvaluate.extend(slicingWindowPredicted*255)

        # Transform the slicingWindowPredicted into a tuple of three dimension!
        if that.save_images:
            threeDimensionSlicingWindow = np.zeros(shape=(slicingWindowPredicted.shape[0],slicingWindowPredicted.shape[1], 3), dtype=np.uint8)

            # with open(logsName, "a+") as img_file:
            #     img_file.write(np.array2string(slicingWindowPredicted))

            for r, _ in enumerate(slicingWindowPredicted):
                for c, pixel in enumerate(slicingWindowPredicted[r]):
                    if pixel >= 0.90: pixel = 1
                    threeDimensionSlicingWindow[r][c] = (pixel*255,)*3
            # Create the image
            imagePredicted[startingX:startingX+constants.getM(), startingY:startingY+constants.getN()] = threeDimensionSlicingWindow

        # if we reach the end of the image, break the while loop.
        if startingX>=constants.IMAGE_WIDTH-constants.getM() and startingY>=constants.IMAGE_HEIGHT-constants.getN():
            break
        # going to the next slicingWindow
        if startingY<constants.IMAGE_HEIGHT-constants.getN(): startingY+=constants.getN()
        else:
            if startingX<constants.IMAGE_WIDTH:
                startingY=0
                startingX+=constants.getM()

    s2 = time.time()
    print("image time: {}".format(round(s2-s1, 3)))
    if that.save_images:
        s1 = time.time()
        # save the image predicted in the specific folder
        cv2.imwrite(that.saveImagesFolder+relativePatientFolder+idx+".png", imagePredicted)
        # HEATMAP
        heatmap_img = cv2.applyColorMap(~imagePredicted, cv2.COLORMAP_JET)
        cv2.imwrite(that.saveImagesFolder+relativePatientFolder+idx+"_heatmap.png", heatmap_img)
        s2 = time.time()
        print("save time: {}".format(round(s2-s1, 3)))
    if that.save_statistics:
        s1 = time.time()
        tn, fn, fp, tp = {}, {}, {}, {}
        for classToEval in that.classes_to_evaluate:
            if classToEval=="penumbra": label=2
            elif classToEval=="core": label=3
            elif classToEval=="penumbracore": label=4
            tn[classToEval], fn[classToEval], fp[classToEval], tp[classToEval] = metrics.mappingPrediction(YTRUEToEvaluate, YPREDToEvaluate, label)

        for func in that.statistics:
            if func.__name__ not in stats.keys(): stats[func.__name__] = {}
            for classToEval in that.classes_to_evaluate:
                if classToEval=="penumbra": label=2
                elif classToEval=="core": label=3
                elif classToEval=="penumbracore": label=4
                if classToEval not in stats[func.__name__].keys(): stats[func.__name__][classToEval] = []

                if func.__name__ == "mAP" or func.__name__ == "AUC" or func.__name__ == "ROC_AUC":
                    res = func(YTRUEToEvaluate, YPREDToEvaluate, label)
                    res = res if not np.isnan(res) else 0
                else:
                    # res = func(tn[classToEval], fn[classToEval], fp[classToEval], tp[classToEval])
                    res = (tn[classToEval], fn[classToEval], fp[classToEval], tp[classToEval])

                # stats[func.__name__][classToEval].append(res)
                stats[func.__name__][classToEval] = res
        s2 = time.time()
        print("stats time: {}".format(round(s2-s1, 3)))
    end = time.time()
    if constants.getVerbose():
        print("Time: {0}s for image {1}.".format(round(end-start, 3), idx))
        general_utils.printSeparation("-", 100)

    return stats

################################################################################
# Test the model with the selected patient
def evaluateModelWithCategorics(nn, p_id):
    sample_weights = nn.getSampleWeights("test")
    suffix = general_utils.getSuffix()

    testing = nn.model.evaluate(
        x=nn.dataset["test"]["data"],
        y=nn.dataset["test"]["labels"],
        sample_weight=sample_weights,
        callbacks=nn.callbacks,
        verbose=constants.getVerbose(),
        use_multiprocessing=nn.mp
    )

    general_utils.printSeparation("-",50)

    for metric_name in nn.train.history:
        print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.printSeparation("-",50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID(p_id)+suffix+".txt", "a+") as text_file:
        for metric_name in nn.train.history:
            text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing):
            text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
        text_file.write("----------------------------------------------------- \n")

################################################################################
# Test the model (already saved) with the selected patient
def evaluateModelAlreadySaved(nn, p_id):
    suffix = general_utils.getSuffix()

    filename_train = nn.datasetFolder+"patient"+str(p_id)+suffix+".hkl"
    nn.train_df = dataset_utils.readFromHickle(filename_train, "")
#    filename_train = nn.datasetFolder+"trainComplete"+str(p_id)+".h5"
#   nn.train_df = dataset_utils.readFromHDF(filename_train, "")
    nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, nn.mp)
    nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, indices=nn.dataset["test"]["indices"])

    nn.compileModel() # compile the model and then evaluate
    sample_weights = nn.getSampleWeights("test")

    testing = nn.model.evaluate(
        x=nn.dataset["test"]["data"],
        y=nn.dataset["test"]["labels"],
        sample_weight=sample_weights,
        callbacks=nn.callbacks,
        verbose=constants.getVerbose(),
        use_multiprocessing=nn.mp
    )

    general_utils.printSeparation("-",50)
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.printSeparation("-",50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID(p_id)+suffix+".txt", "a+") as text_file:
        for index, val in enumerate(testing):
            text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
        text_file.write("----------------------------------------------------- \n")
