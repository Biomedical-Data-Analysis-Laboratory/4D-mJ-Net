import utils, testing
from constants import M,N,SAMPLES,SLICING_PIXELS,PREFIX_IMAGES,getVerbose,getRootPath

import os
from tensorflow.keras.models import model_from_json

################################################################################
# Class that defines a NeuralNetwork
################################################################################
class NeuralNetwork(object):
    """docstring for NeuralNetwork."""

    def __init__(self, info, setting):
        super(NeuralNetwork, self).__init__()

        self.name = info["name"]
        self.epochs = info["epochs"]
        self.optimizer = info["optimizer"]
        self.da = True if info["data_augmentation"]==1 else False
        self.train_again = True if info["train_again"]==1 else False
        self.cross_validation = True if info["cross_validation"]==1 else False
        self.supervised = True if info["supervised"]==1 else False

        self.patientsFolder = setting["relative_paths"]["patients"]
        self.labeledImagesFolder = setting["relative_paths"]["labeled_images"]
        self.datasetFolder = setting["relative_paths"]["dataset"]
        self.savedModelfolder = "SAVE/"+setting["relative_paths"]["save"]["model"]
        self.saveImagesFolder = "SAVE/"+setting["relative_paths"]["save"]["images"]

################################################################################
# return a Boolean to control if the model was already saved
    def isModelSaved(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        return os.path.isfile(saved_modelname) and os.path.isfile(saved_weightname)

################################################################################
# load json and create model
    def loadSavedModel(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)
        json_file =open(saved_modelname, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(saved_weightname)

        if getVerbose():
            utils.printSeparation("+",100)
            print(" --- MODEL {} LOADED FROM DISK! --- ".format(saved_modelname))
            utils.printSeparation("+",100)

        return self.model


################################################################################
# Save the trained model and its relative weights
    def saveModelAndWeight(self, idFunc, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        p_id = getStringPatientIndex(p_id)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filename_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(saved_weightname)

        if constants.getVerbose():
            utils.printSeparation("-", 50)
            print("Saved model and weights to disk!")

################################################################################
# call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, p_id):
        if constants.getVerbose():
            utils.printSeparation("+", 50)
            print("Predicting and saving the images for patient {}".format(p_id))
            utils.printSeparation("+", 50)
        testing.predictAndSaveImages(self, p_id)

################################################################################
# Test the model with the selected patient
    def evaluateModelWithCategorics(self, Y, loss_val, test_labels, training_score, p_id, idFunc):
        if constants.getVerbose():
            utils.printSeparation("+", 50)
            print("Evaluating the model for patient {}".format(p_id))
            utils.printSeparation("+", 50)
        test_labels.evaluateModelWithCategorics(...)
################################################################################
# return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, suffix):
        path = utils.getFullDirectoryPath(self.savedModelfolder)+self.getNNID(p_id)+"_"+str(SLICING_PIXELS)+"_"+str(M)+"x"+str(N)
        return path+suffix

################################################################################
# return the saved model
    def getSavedModel(self, p_id):
        return self.getSavedInformation(p_id, ".json")

################################################################################
# return the saved weight
    def getSavedWeight(self, p_id):
        return self.getSavedInformation(p_id, ".h5")

################################################################################
# return NeuralNetwork ID
    def getNNID(self, p_id):
        id = self.name
        if self.da: id += "_DA"
        id += self.getOptimizerName()
        if self.cross_validation:
            id += ("_" + p_id)

        return id

################################################################################
# return optimizer name
    def getOptimizerName(self):
        return ("_"+self.optimizer["name"].upper())
