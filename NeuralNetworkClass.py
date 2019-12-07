import utils, dataset_utils, models, training, testing, constants

import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical, multi_gpu_model

################################################################################
# Class that defines a NeuralNetwork
################################################################################
class NeuralNetwork(object):
    """docstring for NeuralNetwork."""

    def __init__(self, info, setting):
        super(NeuralNetwork, self).__init__()

        self.name = info["name"]
        self.epochs = info["epochs"]
        self.validation_perc = info["validation_perc"]

        self.dataset = {
            "train": {},
            "val": {},
            "test": {}
        }

        self.optimizer = training.getOptimizer(optInfo=info["optimizer"])

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
# Function to divide the dataframe in train and test based on the patient id;
# plus it reshape the pixel array and initialize the model.
    def prepareDataset(self, train_df, p_id):
        # set the dataset inside the class
        self.train_df = train_df
        if getVerbose():
            utils.printSeparation("+", 50)
            print("Preparing Dataset for patient {}".format(p_id))
            utils.printSeparation("+", 50)

        # get the dataset
        self.dataset = dataset_utils.prepareDataset(self.dataset, self.train_df, self.validation_perc, self.supervised, p_id)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_BRAIN, self.N_CORE, self.N_TOT = getNumberOfElements(self.train_df)

################################################################################
# Run the training over the dataset based on the model
def runTraining(self, p_id, n_gpu):
    self.dataset["train"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=train_df, indices=self.dataset["train"]["indices"])
    self.dataset["val"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=train_df, indices=self.dataset["val"]["indices"])
    if self.supervised: self.dataset["test"]["labels"] = dataset_utils.getLabelsFromIndex(train_df=train_df, indices=self.dataset["test"]["indices"])

    if getVerbose():
        utils.printSeparation("-", 50)
        print(print("[INFO] Getting model {}...".format()))
        utils.printSeparation("-", 50)

    # based on the number of GPUs availables
    # call the function called self.name in models.py
    if n_gpu==1:
        self.model = getattr(models, self.name)(self.dataset["train"]["data"])
    else:
        with tf.device('/cpu:0'):
            self.model = getattr(models, self.name)(self.dataset["train"]["data"])
            self.model = multi_gpu_model(self.model, gpus=n_gpu)

    if getVerbose():
        print(self.model.summary())

    self.model.compile(optimizer=self.optimizer, loss=utils.dice_coef_loss, metrics=[utils.dice_coef])
    class_weights = None
    sample_weights = self.train_df.label.map({
                constants.LABELS[0]:self.N_TOT-self.N_BACKGROUND,
                constants.LABELS[1]:self.N_TOT-self.N_BRAIN,
                constants.LABELS[2]:self.N_TOT-self.N_PENUMBRA,
                constants.LABELS[3]:self.N_TOT-self.N_CORE})
    sample_weights = sample_weights.values[self.dataset["train"]["indices"]]

    # fit and train the model
    self.train = training.fitModel(dataset=self.dataset, epochs=self.epochs, class_weights=class_weights, sample_weights=sample_weights)

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

        if getVerbose():
            utils.printSeparation("-", 50)
            print("Saved model and weights to disk!")
            utils.printSeparation("-", 50)

################################################################################
# Call the function located in testing for predicting and saved the images
    def predictAndSaveImages(self, p_id):
        if getVerbose():
            utils.printSeparation("+", 50)
            print("Predicting and saving the images for patient {}".format(p_id))
            utils.printSeparation("+", 50)
        testing.predictAndSaveImages(self, p_id)

################################################################################
# Test the model with the selected patient
    def evaluateModelWithCategorics(self, Y, loss_val, test_labels, training_score, p_id, idFunc):
        if getVerbose():
            utils.printSeparation("+", 50)
            print("Evaluating the model for patient {}".format(p_id))
            utils.printSeparation("+", 50)

        test_labels.evaluateModelWithCategorics(...)


################################################################################
# return the saved model or weight (based on the suffix)
    def getSavedInformation(self, p_id, suffix):
        path = utils.getFullDirectoryPath(self.savedModelfolder)+self.getNNID(p_id)+"_"+str(constants.SLICING_PIXELS)+"_"+str(M)+"x"+str(N)
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
# return the verbose flag
    def getVerbose(self):
        return constants.getVerbose()

################################################################################
# return optimizer name
    def getOptimizerName(self):
        return ("_"+self.optimizer["name"].upper())
