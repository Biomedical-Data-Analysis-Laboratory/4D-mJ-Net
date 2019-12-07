# Run the testing function, save the images ..

import utils, constants

import os, time
import cv2
import glob
import numpy as np


################################################################################
# Generate the images for the patient and save the images
def predictAndSaveImages(that, p_id):
    relativePatientFolder = constants.PREFIX_IMAGES+p_id+"/"
    patientFolder = that.patientsFolder+relativePatientFolder
    utils.createDir(that.saveImagesFolder+relativePatientFolder)

    if constants.getVerbose():
        utils.printSeparation("-", 10)

    for subfolder in glob.glob(patientFolder+"*/"):
        start = time.time()

        idx = utils.getStringPatientIndex(subfolder.replace(patientFolder, '').replace("/", "")) # image index

        if constants.getVerbose():
            print("Analyzing Patient {0}, image {1}...".format(p_id, idx))

        labeled_image = cv2.imread(that.labeledImagesFolder+"Patient"+p_id+"/"+p_id+idx+".png", 0)
        startingX, startingY = 0, 0
        imagePredicted = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT, 3), dtype=np.uint8)
        testImagePredict = np.zeros(shape=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), dtype=np.uint8)

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

        # # TODO:  multi fucking processing for the image generations

        while True:
            pixels = np.zeros(shape=(constants.NUMBER_OF_IMAGE_PER_SECTION,constants.M,constants.N))
            count = 0
            row, column = 0, 0

            # for each image
            for imagename in np.sort(glob.glob(subfolder+"*.png")):
                filename = imagename.replace(subfolder, '')
                if not that.supervised:
                    image = imagesDict[filename]
                    slicingWindow = utils.getSlicingWindow(image, startingX, startingY, constants.M, constants.N)
                    pixels[count] = slicingWindow
                    count+=1
                else:
                    if filename != "01.png":
                        image = imagesDict[filename]
                        slicingWindow = utils.getSlicingWindow(image, startingX, startingY, constants.M, constants.N)
                        pixels[count] = slicingWindow
                        count+=1

            pixels = pixels.reshape(1, pixels.shape[1], pixels.shape[2], pixels.shape[0], 1)

            ### MODEL PREDICT
            slicingWindowPredicted = that.model.predict(pixels)[0]
            # Transform the slicingWindowPredicted into a touple of three dimension!
            threeDimensionSlicingWindow = np.zeros(shape=(slicingWindowPredicted.shape[0],slicingWindowPredicted.shape[1], 3), dtype=np.uint8)

            for r, _ in enumerate(slicingWindowPredicted):
                for c, pixel in enumerate(slicingWindowPredicted[r]):
                    threeDimensionSlicingWindow[r][c] = (pixel*255,)*3

            # Create the image
            imagePredicted[startingX:startingX+constants.M, startingY:startingY+constants.N] = threeDimensionSlicingWindow
            testImagePredict[startingX:startingX+constants.M, startingY:startingY+constants.N] = slicingWindowPredicted

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
        end = time.time()
        if constants.getVerbose():
            print("Total time: {0}s".format(round(end-start, 3)))
            utils.printSeparation("-", 10)
    if constants.getVerbose():
        utils.printSeparation("-", 10)

################################################################################
# Test the model with the selected patient
def evaluateModelWithCategorics(model, Y, loss_val, test_labels, training_score, p_id, idFunc):
    testing = model.evaluate(Y, test_labels, verbose=VERBOSE)

    loss_testing_score = round(testing[0], 6)
    testing_score = round(testing[1], 6)
    print("-----------------------------------------------------")
    if training_score!=None: print("TRAIN %s: %.2f%%" % (model.metrics_names[1], training_score*100))
    print("TEST %s: %.2f%%" % (model.metrics_names[1], testing_score*100))
    print("-----------------------------------------------------")

    with open(SAVED_TEXT_PATH+str(p_id)+"_"+idFunc+".txt", "a+") as text_file:
        text_file.write("\n -----------------------------------------------------")
        if training_score!=None: text_file.write("\n TRAIN %s: %.2f%%" % (model.metrics_names[1], training_score*100))
        text_file.write("\n TEST %s: %.2f%%" % (model.metrics_names[1], testing_score*100))
        text_file.write("\n -----------------------------------------------------")
        text_file.write("\n TEST LOSS %s: %.2f%%" % (model.metrics_names[1], loss_testing_score))
        text_file.write("\n -----------------------------------------------------")
        text_file.write("\n LOSS: " + str(loss_val))

    return testing_score
