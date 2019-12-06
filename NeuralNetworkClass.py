import utils, testing
from constants import M,N,SAMPLES,SLICING_PIXELS,PREFIX_IMAGES,getVerbose,getRootPath

import os
from tensorflow.keras.models import model_from_json


# Class that defines a NeuralNetwork

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
        self.savedModelfolder = "SAVE/"+setting["relative_paths"]["save"]["model"]
        self.saveImagesFolder = "SAVE/"+setting["relative_paths"]["save"]["images"]


    def isModelSaved(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSAvedWeight(p_id)

        return os.path.isfile(saved_modelname) and os.path.isfile(saved_weightname)

################################################################################
    # load json and create model
    def loadSAvedModel(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSAvedWeight(p_id)
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

    def predictAndSaveImages(self, p_id):
        testing.predictAndSaveImages(self, p_id)

    def getSavedInformation(self, p_id, suffix):
        path = utils.getFullDirectoryPath(self.savedModelfolder)+self.getNNID(p_id)+"_"+str(SLICING_PIXELS)+"_"+str(M)+"x"+str(N)
        return path+suffix

    def getSavedModel(self, p_id):
        return self.getSavedInformation(p_id, ".json")

    def getSAvedWeight(self, p_id):
        return self.getSavedInformation(p_id, ".h5")

    def getNNID(self, p_id):
        id = self.name
        if self.da: id += "_DA"
        id += self.getOptimizerName()
        if self.cross_validation:
            id += ("_" + p_id)

        return id

    def getOptimizerName(self):
        return ("_"+self.optimizer["name"].upper())
