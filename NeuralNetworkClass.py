from Utils import general_utils, dataset_utils
import models, training, testing, constants

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical, multi_gpu_model

################################################################################
# Class that defines a NeuralNetwork
################################################################################
class NeuralNetwork(object):
    """docstring for NeuralNetwork."""

    def __init__(self, info, setting):
        super(NeuralNetwork, self).__init__()
        self.summaryFlag = 0

        self.name = info["name"]
        self.epochs = info["epochs"]
        self.validation_perc = info["validation_perc"]

        self.dataset = {
            "train": {},
            "val": {},
            "test": {}
        }

        self.optimizerName = info["optimizer"]["name"]
        self.loss = general_utils.getLoss(info["loss"])

        self.da = True if info["data_augmentation"]==1 else False
        self.train_again = True if info["train_again"]==1 else False
        self.cross_validation = True if info["cross_validation"]==1 else False
        self.supervised = True if info["supervised"]==1 else False

        self.rootPath = setting["root_path"]
        self.datasetFolder = setting["dataset_path"]
        self.patientsFolder = setting["relative_paths"]["patients"]
        self.labeledImagesFolder = setting["relative_paths"]["labeled_images"]
        self.savedModelfolder = "SAVE/"+setting["relative_paths"]["save"]["model"]
        self.saveImagesFolder = "SAVE/"+setting["relative_paths"]["save"]["images"]
        self.savePlotFolder = "SAVE/"+setting["relative_paths"]["save"]["plot"]
        self.saveTextFolder = "SAVE/"+setting["relative_paths"]["save"]["text"]

        self.optimizer = training.getOptimizer(optInfo=info["optimizer"])
        self.infoCallbacks = info["callbacks"]

################################################################################
# Initialize the callbacks
    def setCallbacks(self, p_id):
        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("Setting callbacks...")
            general_utils.printSeparation("-", 50)
        self.callbacks = training.getCallbacks(root_path=self.rootPath, info=self.infoCallbacks, filename=self.getSavedInformation(p_id))

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

        if self.getVerbose():
            general_utils.printSeparation("+",100)
            print(" --- MODEL {} LOADED FROM DISK! --- ".format(saved_modelname))
            print(" --- WEIGHTS {} LOADED FROM DISK! --- ".format(saved_weightname))
            general_utils.printSeparation("+",100)

        return self.model

################################################################################
# Check if there are saved partial weights
    def arePartialWeightsSaved(self, p_id):
        self.partialWeightsPath = ""
        # path ==> weight name plus a suffix ":"
        path = self.getSavedInformation(p_id)+":"
        for file in glob.glob(self.savedModelfolder+"*.h5"):
            if path in self.rootPath+file: # we have a match
                self.partialWeightsPath = file
                return True

        return False

################################################################################
# Load the partial weights and set the initial epoch where the weights were saved
    def loadModelFromPartialWeights(self, p_id):
        if self.partialWeightsPath!="":
            self.model.load_weights(self.partialWeightsPath)
            self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partialWeightsPath)

            if self.getVerbose():
                general_utils.printSeparation("+",100)
                print(" --- WEIGHTS {} LOADED FROM DISK! --- ".format(self.partialWeightsPath))
                print(" --- Start training from epoch {} --- ".format(str(self.initial_epoch)))
                general_utils.printSeparation("+",100)

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
    def prepareDataset(self, train_df, p_id):
        # set the dataset inside the class
        self.train_df = train_df
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Preparing Dataset for patient {}...".format(p_id))
            general_utils.printSeparation("+", 50)

        # get the dataset
        self.dataset = dataset_utils.prepareDataset(self.dataset, self.train_df, self.validation_perc, self.supervised, p_id)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = general_utils.getNumberOfElements(self.train_df)

################################################################################
# Run the training over the dataset based on the model
    def runTraining(self, p_id, n_gpu):
        self.dataset["train"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["train"]["indices"])
        self.dataset["val"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["val"]["indices"])
        if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["test"]["indices"])

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] Getting model {0} with {1} optimizer...".format(self.name, self.optimizerName))
            general_utils.printSeparation("-", 50)

        # based on the number of GPUs availables
        # call the function called self.name in models.py
        if n_gpu==1:
            self.model = getattr(models, self.name)(self.dataset["train"]["data"])
        else:
            with tf.device('/cpu:0'):
                self.model = getattr(models, self.name)(self.dataset["train"]["data"])
                self.model = multi_gpu_model(self.model, gpus=n_gpu)

        if self.getVerbose() and self.summaryFlag==0:
            print(self.model.summary())
            self.summaryFlag+=1

        # check if the model has some saved weights to load...
        self.initial_epoch = 0
        if self.arePartialWeightsSaved(p_id):
            self.loadModelFromPartialWeights(p_id)

        self.model.compile(optimizer=self.optimizer, loss=self.loss["loss"], metrics=[self.loss["metrics"]])

        class_weights = {
            constants.LABELS.index("background"):(self.N_TOT-self.N_BACKGROUND)/self.N_TOT,
            constants.LABELS.index("brain"):(self.N_TOT-self.N_BRAIN)/self.N_TOT,
            constants.LABELS.index("penumbra"):(self.N_TOT-self.N_PENUMBRA)/self.N_TOT,
            constants.LABELS.index("core"):(self.N_TOT-self.N_CORE)/self.N_TOT}

        sample_weights = self.train_df.label.map({
                    constants.LABELS[0]:self.N_TOT-self.N_BACKGROUND,
                    constants.LABELS[1]:self.N_TOT-self.N_BRAIN,
                    constants.LABELS[2]:self.N_TOT-self.N_PENUMBRA,
                    constants.LABELS[3]:self.N_TOT-self.N_CORE})
        sample_weights = sample_weights.values[self.dataset["train"]["indices"]]

        # fit and train the model
        self.train = training.fitModel(
                model=self.model,
                dataset=self.dataset,
                epochs=self.epochs,
                listOfCallbacks=self.callbacks,
                class_weights=class_weights,
                sample_weights=sample_weights,
                initial_epoch=self.initial_epoch)

        self.training_score = round(self.train.history[self.loss["name"]][num_epochs-1], 6)
        self.loss_val = round(self.train.history["loss"][num_epochs-1], 6)

        # plot the loss and accuracy of the training
        training.plotLossAndAccuracy(self, p_id)

################################################################################
# Save the trained model and its relative weights
    def saveModelAndWeight(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        p_id = general_utils.getStringPatientIndex(p_id)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filename_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(saved_weightname)

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] Saved model and weights to disk!")
            general_utils.printSeparation("-", 50)

################################################################################
# Call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, p_id):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Predicting and saving the images for patient {}".format(p_id))
            general_utils.printSeparation("+", 50)

        testing.predictAndSaveImages(self, p_id)

################################################################################
# Test the model with the selected patient
    def evaluateModelWithCategorics(self, p_id):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Evaluating the model for patient {}".format(p_id))
            general_utils.printSeparation("+", 50)

        self.testing_score = testing.evaluateModelWithCategorics(self)


################################################################################
# return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, suffix=""):
        # mJ-Net_DA_ADAM_4_16x16.json <-- example weights name
        # mJ-Net_DA_ADAM_4_16x16.h5 <-- example model name
        path = general_utils.getFullDirectoryPath(self.savedModelfolder)+self.getNNID(p_id)+"_"+str(constants.SLICING_PIXELS)+"_"+str(constants.M)+"x"+str(constants.N)
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
        id += ("_"+self.optimizerName.upper())
        if self.cross_validation:
            id += ("_" + p_id)

        return id

################################################################################
# return the verbose flag
    def getVerbose(self):
        return constants.getVerbose()
