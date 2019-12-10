# Run the testing function, save the images ..
from Utils import general_utils, dataset_utils
import constants

import os, time, cv2, glob
import multiprocessing
import numpy as np


def predictFromModel(nn, input):
    return nn.model.predict(
            x=input,
            steps=nn.test_steps,
            use_multiprocessing=nn.mp
    )

################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(that, p_id):
    start = time.time()

    relativePatientFolder = constants.PREFIX_IMAGES+p_id+"/"
    patientFolder = that.patientsFolder+relativePatientFolder
    general_utils.createDir(that.saveImagesFolder+relativePatientFolder)

    if constants.getVerbose():
        general_utils.printSeparation("-", 100)
        general_utils.printSeparation("-", 100)

    # if that.mp:
    #     cpu_count = multiprocessing.cpu_count()
    #     input = []
    #     for subfolder in glob.glob(patientFolder+"*/"):
    #         input.append((that, subfolder, p_id, patientFolder, relativePatientFolder))
    #     with multiprocessing.Pool(processes=cpu_count) as pool: # auto closing workers
    #         pool.starmap(predictImage, input)
    # else:
    for subfolder in glob.glob(patientFolder+"*/"):
        predictImage(that, subfolder, p_id, patientFolder, relativePatientFolder)

    end = time.time()
    print("Total time: {0}s for patient {1}.".format(round(end-start, 3), p_id))
    if constants.getVerbose():
        general_utils.printSeparation("-", 100)

################################################################################
# Generate a SINGLE image for the patient and save it
def predictImage(that, subfolder, p_id, patientFolder, relativePatientFolder):
    start = time.time()

    idx = general_utils.getStringPatientIndex(subfolder.replace(patientFolder, '').replace("/", "")) # image index

    if constants.getVerbose():
        print("Analyzing Patient {0}, image {1}...".format(p_id, idx))

    labeled_image = cv2.imread(that.labeledImagesFolder+"Patient"+p_id+"/"+p_id+idx+".png", 0)
    startingX, startingY = 0, 0
    imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, 3), dtype=np.uint8)

    imagesDict = {} # faster access to the images
    for imagename in np.sort(glob.glob(subfolder+"*.png")): # sort the images !
        filename = imagename.replace(subfolder, '')
        if not that.supervised:
            image = cv2.imread(imagename, 0)
            imagesDict[filename] = image
        else:
            if filename != "01.png": # don't take the first image (the manually annotated one)
                image = cv2.imread(imagename, 0)
                imagesDict[filename] = image

    # Generate the predicted image
    while True:
        pixels = np.zeros(shape=(constants.NUMBER_OF_IMAGE_PER_SECTION,constants.M,constants.N))
        count = 0
        row, column = 0, 0

        # for each image
        for imagename in np.sort(glob.glob(subfolder+"*.png")):
            filename = imagename.replace(subfolder, '')
            if not that.supervised:
                image = imagesDict[filename]
                pixels[count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.M, constants.N)
                count+=1
            else:
                if filename != "01.png":
                    image = imagesDict[filename]
                    pixels[count] = general_utils.getSlicingWindow(image, startingX, startingY, constants.M, constants.N)
                    count+=1

        pixels = pixels.reshape(1, pixels.shape[1], pixels.shape[2], pixels.shape[0], 1)

        ### MODEL PREDICT
        slicingWindowPredicted = predictFromModel(that, pixels)[that.test_steps-1]
        # slicingWindowPredicted contain only the prediction for the last step

        # Transform the slicingWindowPredicted into a touple of three dimension!
        threeDimensionSlicingWindow = np.zeros(shape=(slicingWindowPredicted.shape[0],slicingWindowPredicted.shape[1], 3), dtype=np.uint8)

        for r, _ in enumerate(slicingWindowPredicted):
            for c, pixel in enumerate(slicingWindowPredicted[r]):
                threeDimensionSlicingWindow[r][c] = (pixel*255,)*3

        # Create the image
        imagePredicted[startingX:startingX+constants.M, startingY:startingY+constants.N] = threeDimensionSlicingWindow

        if startingX>=constants.IMAGE_WIDTH-constants.M and startingY>=constants.IMAGE_HEIGHT-constants.N: # if we reach the end of the image, break the while loop.
            break
        # going to the next slicingWindow
        if startingY<constants.IMAGE_HEIGHT-constants.N: startingY+=constants.N
        else:
            if startingX<constants.IMAGE_WIDTH:
                startingY=0
                startingX+=constants.M

    # save the image predicted in the specific folder
    cv2.imwrite(that.saveImagesFolder+relativePatientFolder+idx+".png", imagePredicted)

    # HEATMAP
    heatmap_img = cv2.applyColorMap(imagePredicted, cv2.COLORMAP_JET)
    cv2.imwrite(that.saveImagesFolder+relativePatientFolder+idx+"_heatmap2.png", heatmap_img)

    end = time.time()
    if constants.getVerbose():
        print("Time: {0}s for image {1}.".format(round(end-start, 3), idx))
        general_utils.printSeparation("-", 100)

################################################################################
# Test the model with the selected patient
def evaluateModelWithCategorics(nn, p_id, mp):
    testing = nn.model.evaluate(nn.dataset["test"]["data"], nn.dataset["test"]["labels"], verbose=constants.getVerbose(), use_multiprocessing=mp)

    loss_testing_score = round(testing[0], 6)
    testing_score = round(testing[1], 6)
    general_utils.printSeparation("-",50)
    if nn.training_score!=None: print("TRAIN %s: %.2f%%" % (nn.model.metrics_names[1], nn.training_score*100))
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
    general_utils.printSeparation("-",50)

    with open(general_utils.getFullDirectoryPath(nn.saveTextFolder)+nn.getNNID(p_id)+".txt", "a+") as text_file:
        text_file.write("\n -----------------------------------------------------")
        if training_score!=None: text_file.write("\n TRAIN %s: %.2f%%" % (nn.model.metrics_names[1], nn.training_score*100))
        text_file.write("\n -----------------------------------------------------")
        text_file.write("\n LOSS: " + str(nn.loss_val))
        text_file.write("\n -----------------------------------------------------")
        text_file.write("\n TEST %s: %.2f%%" % (nn.model.metrics_names[1], testing_score*100))
        text_file.write("\n -----------------------------------------------------")
        text_file.write("\n TEST LOSS %s: %.2f%%" % (nn.model.metrics_names[1], loss_testing_score))

    return testing_score

################################################################################

def evaluateModelAlreadySaved(nn, p_id, mp):
    filename_train = nn.datasetFolder+"trainComplete"+str(p_id)+".h5"
    nn.train_df = dataset_utils.readFromHDF(filename_train, "")
    nn.dataset = dataset_utils.getTestDataset(nn.dataset, nn.train_df, p_id, mp)
    nn.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=nn.train_df, indices=nn.dataset["test"]["indices"])

    nn.compileModel() # compile the model and then evaluate
    testing = nn.model.evaluate(nn.dataset["test"]["data"], nn.dataset["test"]["labels"], verbose=constants.getVerbose(), use_multiprocessing=mp)

    general_utils.printSeparation("-",50)
    for index, val in enumerate(testing):
        print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
