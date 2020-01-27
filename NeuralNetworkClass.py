from Utils import general_utils, dataset_utils, models
import training, testing, constants

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

        # Used to override the path for the saved model in order to test patients with a specific model
        self.OVERRIDE_MODELS_ID_PATH = setting["OVERRIDE_MODELS_ID_PATH"] if setting["OVERRIDE_MODELS_ID_PATH"]!="" else False

        self.name = info["name"]
        self.epochs = info["epochs"]
        self.val = {
            "validation_perc": info["val"]["validation_perc"],
            "random_validation_selection": info["val"]["random_validation_selection"]
            }
        self.test_steps = info["test_steps"]

        self.dataset = {
            "train": {},
            "val": {},
            "test": {}
            }

        self.optimizerInfo = info["optimizer"]
        self.params = info["params"]
        self.loss = general_utils.getLoss(info["loss"])
        self.classes_to_evaluate = info["classes_to_evaluate"]
        self.metricFuncs = general_utils.getStatisticFunctions(info["metrics"])
        self.statistics = general_utils.getStatisticFunctions(info["statistics"])

        self.da = True if info["data_augmentation"]==1 else False
        self.train_again = True if info["train_again"]==1 else False
        self.cross_validation = True if info["cross_validation"]==1 else False
        self.supervised = True if info["supervised"]==1 else False

        self.rootPath = setting["root_path"]
        self.datasetFolder = setting["dataset_path"]
        self.patientsFolder = setting["relative_paths"]["patients"]
        self.labeledImagesFolder = setting["relative_paths"]["labeled_images"]
        self.savedModelFolder = "SAVE/"+setting["relative_paths"]["save"]["model"]
        self.savePartialModelFolder = "SAVE/"+setting["relative_paths"]["save"]["partial_model"]
        self.saveImagesFolder = "SAVE/"+setting["relative_paths"]["save"]["images"]
        self.savePlotFolder = "SAVE/"+setting["relative_paths"]["save"]["plot"]
        self.saveTextFolder = "SAVE/"+setting["relative_paths"]["save"]["text"]

        self.infoCallbacks = info["callbacks"]

################################################################################
# Initialize the callbacks
    def setCallbacks(self, p_id, sample_weights=None):
        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("Setting callbacks...")

        self.callbacks = training.getCallbacks(
            root_path=self.rootPath,
            info=self.infoCallbacks,
            filename=self.getSavedInformation(p_id, path=self.savePartialModelFolder),
            textFolderPath=self.saveTextFolder,
            dataset=self.dataset,
            #model=self.model,
            sample_weights=sample_weights
        )

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

################################################################################
# Check if there are saved partial weights
    def arePartialWeightsSaved(self, p_id):
        self.partialWeightsPath = ""
        # path ==> weight name plus a suffix ":" <-- constants.suffix_partial_weights
        path = self.getSavedInformation(p_id, path=self.savePartialModelFolder)+constants.suffix_partial_weights
        for file in glob.glob(self.savePartialModelFolder+"*.h5"):
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

################################################################################
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
    def prepareDataset(self, train_df, p_id):
        # set the dataset inside the class
        self.train_df = train_df
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Preparing Dataset for patient {}...".format(p_id))

        # get the dataset
        self.dataset = dataset_utils.prepareDataset(self, p_id)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.getNumberOfElements(self.train_df)

################################################################################
# compile the model, callable also from outside
    def compileModel(self):
        # set the optimizer (or reset)
        self.optimizer = training.getOptimizer(optInfo=self.optimizerInfo)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss["loss"],
            metrics=[self.metricFuncs]
        )

################################################################################
# Run the training over the dataset based on the model
    def runTraining(self, p_id, n_gpu):
        self.dataset["train"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["train"]["indices"])
        self.dataset["val"]["labels"] = None if self.val["validation_perc"]==0 else dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["val"]["indices"])
        if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=self.train_df, indices=self.dataset["test"]["indices"])

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] Getting model {0} with {1} optimizer...".format(self.name, self.optimizerInfo["name"]))

        # based on the number of GPUs availables
        # call the function called self.name in models.py
        if n_gpu==1:
            self.model = getattr(models, self.name)(self.dataset["train"]["data"], params=self.params)
        else:
            # TODO: problems during the load of the model (?)
            with tf.device('/cpu:0'):
                self.model = getattr(models, self.name)(self.dataset["train"]["data"], params=self.params)
            self.model = multi_gpu_model(self.model, gpus=n_gpu)

        if self.getVerbose() and self.summaryFlag==0:
            print(self.model.summary())
            self.summaryFlag+=1

        # check if the model has some saved weights to load...
        self.initial_epoch = 0
        if self.arePartialWeightsSaved(p_id):
            self.loadModelFromPartialWeights(p_id)

        # compile the model with optimizer, loss function and metrics
        self.compileModel()

        sample_weights = self.getSampleWeights("train")

        # Set the callbacks
        self.setCallbacks(p_id, sample_weights)

        # fit and train the model
        self.train = training.fitModel(
                model=self.model,
                dataset=self.dataset,
                epochs=self.epochs,
                listOfCallbacks=self.callbacks,
                sample_weights=sample_weights,
                initial_epoch=self.initial_epoch,
                use_multiprocessing=self.mp)

        # plot the loss and accuracy of the training
        training.plotLossAndAccuracy(self, p_id)

################################################################################
# Get the sample weight from the dataset
    def getSampleWeights(self, flag):

        sample_weights = self.train_df.label.map({
            constants.LABELS[0]:1,
            constants.LABELS[1]:1,
            constants.LABELS[2]:10,
            constants.LABELS[3]:100
        })

        return sample_weights.values[self.dataset[flag]["indices"]]

################################################################################
# Save the trained model and its relative weights
    def saveModelAndWeight(self, p_id):
        saved_modelname = self.getSavedModel(p_id)
        saved_weightname = self.getSavedWeight(p_id)

        p_id = general_utils.getStringPatientIndex(p_id)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(saved_modelname, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(saved_weightname)

        if self.getVerbose():
            general_utils.printSeparation("-", 50)
            print("[INFO] Saved model and weights to disk!")

################################################################################
# Call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, p_id, stats):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Predicting and saving the images for patient {}".format(p_id))

        stats = testing.predictAndSaveImages(self, p_id, stats)
        self.saveStats(stats, p_id)
        return stats

################################################################################
# Function to save in a file the statistic for the test patients
    def saveStats(self, stats, p_id):
        suffix = general_utils.getSuffix()
        with open(general_utils.getFullDirectoryPath(self.saveTextFolder)+self.getNNID(p_id)+suffix+".txt", "a+") as text_file:
            for func in self.statistics:
                for classToEval in self.classes_to_evaluate:
                    meanV = np.mean(stats[func.__name__][classToEval])
                    stdV = np.std(stats[func.__name__][classToEval])
                    text_file.write("TEST MEAN %s %s: %.2f%% \n" % (func.__name__, classToEval, round(meanV,6)*100))
                    text_file.write("TEST STD %s %s: %.2f \n" % (func.__name__, classToEval, round(stdV,6)))
                    text_file.write("+++++ \n")
                text_file.write("----------------------------------------------------- \n")

################################################################################
# Test the model with the selected patient
    def evaluateModelWithCategorics(self, p_id, isAlreadySaved):
        if self.getVerbose():
            general_utils.printSeparation("+", 50)
            print("Evaluating the model for patient {}".format(p_id))

        if isAlreadySaved:
            self.testing_score = testing.evaluateModelAlreadySaved(self, p_id)
        else:
            self.testing_score = testing.evaluateModelWithCategorics(self, p_id)

################################################################################
# set the flag for single/multi PROCESSING
    def setProcessingEnv(self, mp):
        self.mp = mp

################################################################################
# return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, path, other_info="", suffix=""):
        # mJ-Net_DA_ADAM_4_16x16.json <-- example weights name
        # mJ-Net_DA_ADAM_4_16x16.h5 <-- example model name
        path = general_utils.getFullDirectoryPath(path)+self.getNNID(p_id)+other_info+general_utils.getSuffix()
        return path+suffix

################################################################################
# return the saved model
    def getSavedModel(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".json")

################################################################################
# return the saved weight
    def getSavedWeight(self, p_id):
        return self.getSavedInformation(p_id, path=self.savedModelFolder, suffix=".h5")

################################################################################
# return NeuralNetwork ID
    def getNNID(self, p_id):
        # CAREFUL WITH THIS
        if self.OVERRIDE_MODELS_ID_PATH:
            # needs to override the model id to use a different model to test various patients
            id = self.OVERRIDE_MODELS_ID_PATH
        else:
            id = self.name
            if self.da: id += "_DA"
            id += ("_"+self.optimizerInfo["name"].upper())

            id += ("_VAL"+str(self.val["validation_perc"]))
            if self.val["random_validation_selection"]: id += ("_RANDOM")

            # if there is cross validation, add the PATIENT_ID to differenciate the models
            if self.cross_validation:
                id += ("_" + p_id)

        return id

################################################################################
# return the verbose flag
    def getVerbose(self):
        return constants.getVerbose()
