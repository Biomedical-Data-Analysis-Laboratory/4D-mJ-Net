import warnings

from Utils import general_utils, dataset_utils, sequence_utils, architectures, model_utils
from Model import training, testing
from Model.constants import *

import os, glob, math, cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model  # multi_gpu_model
from tcn.tcn.tcn import TCN

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Class that defines a NeuralNetwork
################################################################################

class NeuralNetwork(object):
    """docstring for NeuralNetwork."""
    def __init__(self, model_info, setting):
        super(NeuralNetwork, self).__init__()

        # Used to override the path for the saved model in order to test patients with a specific model
        self.OVERRIDE_MODELS_ID_PATH = setting["OVERRIDE_MODELS_ID_PATH"] if setting["OVERRIDE_MODELS_ID_PATH"]!="" else False
        self.use_sequence = setting["USE_SEQUENCE_TRAIN"] if "USE_SEQUENCE_TRAIN" in setting.keys() else 0

        self.name = model_info["name"]
        self.epochs = model_info["epochs"]
        self.batch_size = model_info["batch_size"] if "batch_size" in model_info.keys() else 32

        # use only for the fit_generator
        self.steps_per_epoch_ratio = model_info["steps_per_epoch_ratio"] if "steps_per_epoch_ratio" in model_info.keys() else 1
        self.validation_steps_ratio = model_info["validation_steps_ratio"] if "validation_steps_ratio" in model_info.keys() else 1
        self.back_perc = model_info["back_perc"] if "back_perc" in model_info.keys() else 1

        self.val = {
            "validation_perc": model_info["val"]["validation_perc"],
            "random_validation_selection": model_info["val"]["random_validation_selection"],
            "number_patients_for_validation": model_info["val"]["number_patients_for_validation"] if "number_patients_for_validation" in model_info["val"].keys() else 0,
            "number_patients_for_testing": model_info["val"]["number_patients_for_testing"] if "number_patients_for_testing" in model_info["val"].keys() else 0,
            "seed": model_info["val"]["seed"]
        }

        self.cross_validation = model_info["cross_validation"]

        self.dataset = {"train": {},"val": {},"test": {}}

        # get parameter for the model
        self.optimizer_info = model_info["optimizer"]
        self.params = model_info["params"]
        self.loss = general_utils.get_loss(model_info)
        self.metric_func = general_utils.getMetricFunctions(model_info["metrics"])

        # inflate and concatenate only work with PMs_segmentation architectures!
        for key in ["concatenate_input","convertImgToGray","inflate_network"]:
            self.params[key] = True if key in self.params.keys() and self.params[key] else False

        # FLAGS for the model
        self.input_img_flag = cv2.IMREAD_COLOR if not self.params["convertImgToGray"] else cv2.IMREAD_GRAYSCALE  # only works when the input are the PMs (concatenate)
        self.multi_input = self.params["multiInput"] if "multiInput" in self.params.keys() else dict()

        self.model_info = dict()
        for key in ["to_categ","save_images","data_augmentation","supervised","save_activation_filter","use_hickle","SVO_focus","MONTE_CARLO_DROPOUT"]:
            self.model_info[key] = True if key in model_info.keys() and model_info[key] ==1 else False

        set_TO_CATEG(self.model_info["to_categ"])

        self.is3dot5DModel = True if "3dot5D" in self.name else False
        self.is4DModel = True if "4D" in self.name else False

        # paths
        self.rootpath = setting["root_path"]
        self.ds_folder = setting["dataset_path"]
        self.labeled_img_folder = setting["relative_paths"]["labeled_images"] if "labeled_images" in setting["relative_paths"].keys() else ""
        self.patients_folder = setting["relative_paths"]["patients"]
        self.experimentID = "EXP" + general_utils.convert_expnum_to_str(setting["EXPERIMENT"])
        self.experiment_folder = "SAVE/" + self.experimentID + "/"
        self.saved_model_folder = self.experiment_folder + setting["relative_paths"]["save"]["model"]
        self.save_partial_model_folder = self.experiment_folder + setting["relative_paths"]["save"]["partial_model"]
        self.save_img_folder = self.experiment_folder + setting["relative_paths"]["save"]["images"]
        self.save_plot_folder = self.experiment_folder + setting["relative_paths"]["save"]["plot"]
        self.save_text_folder = self.experiment_folder + setting["relative_paths"]["save"]["text"]
        self.intermediate_activation_folder = self.experiment_folder + setting["relative_paths"]["save"]["intermediate_activation"] if "intermediate_activation" in setting["relative_paths"]["save"].keys() else None

        self.info_callbacks = model_info["callbacks"]

        # empty variables initialization
        self.n_slices = 0 if "n_slices" not in self.params.keys() else self.params["n_slices"]
        self.x_label = "pixels" if not get_USE_PM() else get_list_PMS()
        self.y_label = "ground_truth"
        self.summary_flag = 0
        self.model = None
        self.model_split = None
        self.testing_score = []
        self.partial_weights_path = ""
        self.callbacks = None
        self.initial_epoch = 0
        self.train_df, self.val_list, self.test_list = None, None, None
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = 0, 0, 0, 0, 0
        self.optimizer = None
        self.sample_weights = None
        self.train = None
        self.train_sequence, self.val_sequence = None, None
        self.mp = False
        self.mp_in_nn = False

        # change the prefix if SUS2020_v2 is in the dataset name
        if "SUS2020" in self.ds_folder: set_prefix("CTP_")

    ################################################################################
    # Set model ID
    def reset_vars(self):
        self.dataset = {"train": {}, "val": {}, "test": {}}

        # empty variables initialization
        self.n_slices = 0 if "n_slices" not in self.params.keys() else self.params["n_slices"]
        self.x_label = "pixels" if not get_USE_PM() else get_list_PMS()
        self.y_label = "ground_truth"
        self.summary_flag = 0
        self.model = None
        self.model_split = None
        self.testing_score = []
        self.partial_weights_path = ""
        self.callbacks = None
        self.initial_epoch = 0
        self.train_df, self.val_list, self.test_list = None, None, None
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = 0, 0, 0, 0, 0
        self.optimizer = None
        self.sample_weights = None
        self.train = None
        self.train_sequence, self.val_sequence = None, None
        self.mp = False
        self.mp_in_nn = False

    ################################################################################
    # Set model ID
    def set_model_split(self, model_split): self.model_split = model_split

    ################################################################################
    # Initialize the callbacks
    def set_callbacks(self, sample_weights=None, add_for_finetuning=""):
        if is_verbose():
            general_utils.print_sep("-", 50)
            print("[INFO] - Setting callbacks...")

        self.callbacks = training.get_callbacks(info=self.info_callbacks, root_path=self.rootpath,
                                                filename=self.getSavedInformation(path=self.save_partial_model_folder),
                                                text_fold_path=self.save_text_folder, dataset=self.dataset,
                                                sample_weights=sample_weights, nn_id=self.get_nn_id(),
                                                add_for_finetuning=add_for_finetuning)

    ################################################################################
    # return a Boolean to control if the model was already saved
    def is_model_saved(self):
        saved_modelname = self.get_saved_model()
        saved_weightname = self.get_saved_weights()

        return os.path.isfile(saved_modelname) and os.path.isfile(saved_weightname)

    ################################################################################
    # load json and create model
    def load_saved_model(self):
        saved_modelname = self.get_saved_model()
        saved_weightname = self.get_saved_weights()
        json_file = open(saved_modelname, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json, custom_objects={"TCN": TCN, "MonteCarloDropout": model_utils.MonteCarloDropout})
        # load weights into new model
        self.model.load_weights(saved_weightname)

        if is_verbose():
            general_utils.print_sep("+", 100)
            print("[INFO - Loading] - --- MODEL {} LOADED FROM DISK! --- ".format(saved_modelname))
            print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(saved_weightname))

    ################################################################################
    # Check if there are saved partial weights
    def are_partial_weights_saved(self):
        # path ==> weight name plus a suffix ":" <-- suffix_partial_weights
        path = self.getSavedInformation(path=self.save_partial_model_folder) + suffix_partial_weights
        for file in glob.glob(self.save_partial_model_folder + "*.h5"):
            if path in self.rootpath+file:  # we have a match
                self.partial_weights_path = file
                return True

        return False

    ################################################################################
    # Load the partial weights and set the initial epoch where the weights were saved
    def load_model_from_partial_weights(self):
        if self.partial_weights_path!= "":
            self.model.load_weights(self.partial_weights_path)
            self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partial_weights_path)

            if is_verbose():
                general_utils.print_sep("+", 100)
                print("[INFO - Loading] - --- WEIGHTS {} LOADED FROM DISK! --- ".format(self.partial_weights_path))
                print("[INFO] - --- Start training from epoch {} --- ".format(str(self.initial_epoch)))

    ################################################################################
    # Function to divide the dataframe in train and test based on the patient id;
    def split_ds(self, train_df, patientlist_train_val, patientlist_test):
        # set the dataset inside the class
        self.train_df = train_df
        # split the dataset (return in dataset just the indices and labels)
        self.dataset, self.val_list, self.test_list = dataset_utils.split_ds(self, patientlist_train_val, patientlist_test)
        # get the number of element per class in the dataset
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.get_number_of_elem(self.train_df)

        return self.val_list

    ################################################################################
    # Function to reshape the pixel array and initialize the model.
    def prepare_ds(self):
        # split the dataset (set the data key inside dataset [NOT for the sequence generator])
        self.dataset = dataset_utils.prepare_ds(self)

    ################################################################################
    # compile the model, callable also from outside
    def compile_model(self):
        # set the optimizer (or reset)
        self.optimizer = training.get_optimizer(opt_info=self.optimizer_info)
        # Compile the model with optimizer, loss, and metrics
        self.model.compile(optimizer=self.optimizer, loss=self.loss["loss"], metrics=self.metric_func)

    ################################################################################
    # Function to route the right initialization and training
    def init_and_start_training(self, n_gpu, jump):
        if self.use_sequence:  # if we are doing a sequence train (for memory issue)
            self.prepare_sequence_class()
            self.init_training(n_gpu)
            if not jump: self.run_train_sequence()
            self.gradual_fine_tuning_solution()
            training.plot_loss_and_accuracy(self)  # Plot the loss and accuracy of the training
        else:
            self.prepare_ds()  # PREPARE DATASET (=divide in train/val/test)
            self.run_training(n_gpu)  # SET THE CALLBACKS, RUN TRAINING & SAVE THE MODELS WEIGHTS
        self.save_model_and_weights()

    ################################################################################
    # Function that initialize the training, print the model summary and set the weights
    def init_training(self, n_gpu):
        assert n_gpu==1, "The number of GPU should be 1."

        if is_verbose():
            general_utils.print_sep("*", 50)
            print("[INFO] - Start runTraining function.")
            print("[INFO] - Getting model {0} with: {1} and {2}".format(self.name, self.optimizer_info["name"], self.loss["name"]))

        # Based on the number of GPUs available, call the function called self.name in architectures.py
        self.model = getattr(architectures, self.name)(params=self.params, multi_input=self.multi_input)

        if self.summary_flag==0:
            for rankdir in ["TB"]:  # "LR"
                plot_model(
                    self.model,
                    to_file=general_utils.get_dir_path(self.saved_model_folder)+self.get_nn_id()+"_"+rankdir+".png",
                    show_shapes=True,
                    rankdir=rankdir)
            self.summary_flag+=1

        memUsage = general_utils.get_model_memory_usage(self.model, self.batch_size)
        if is_verbose(): print("The memory usage for the model is: {}Gb".format(memUsage))
        # Check if the model has some saved weights to load...
        if self.are_partial_weights_saved(): self.load_model_from_partial_weights()

        self.compile_model()  # Compile the model with optimizer, loss function, and metrics
        self.sample_weights = self.get_sample_weights("train")  # Get the sample weights for training
        self.set_callbacks(self.sample_weights)

    ################################################################################
    # Run the training over the dataset based on the model
    def run_training(self, n_gpu):
        self.init_training(n_gpu)
        # Get the labels for training, validation, and testing
        self.dataset["train"]["labels"] = dataset_utils.get_labels_from_idx(train_df=self.train_df,dataset=self.dataset["train"],modelname=self.name, flag="train")
        self.dataset["val"]["labels"] = None if self.val["validation_perc"]==0 else dataset_utils.get_labels_from_idx(train_df=self.train_df, dataset=self.dataset["val"], modelname=self.name, flag="val")
        if self.model_info["supervised"]: self.dataset["test"]["labels"] = dataset_utils.get_labels_from_idx(train_df=self.train_df,dataset=self.dataset["test"],modelname=self.name,flag="test")
        # Train the model
        self.train = training.fit_model(model=self.model, dataset=self.dataset, batch_size=self.batch_size,
                                        epochs=self.epochs, callbacklist=self.callbacks,
                                        sample_weights=self.sample_weights, initial_epoch=self.initial_epoch,
                                        use_multiprocessing=self.mp_in_nn)

        # plot the loss and accuracy of the training
        training.plot_loss_and_accuracy(self)

        # deallocate memory
        for flag in ["train", "val", "test"]:
            for t in ["labels", "data"]:
                if t in self.dataset[flag]: del self.dataset[flag][t]

    ################################################################################
    # Function to prepare the train and validation sequence using the datasetSequence class
    def prepare_sequence_class(self):
        # train data sequence
        self.train_sequence = sequence_utils.ds_sequence(
            dataframe=self.train_df,
            indices=self.dataset["train"]["indices"],
            sample_weights=self.get_sample_weights("train"),
            x_label=self.x_label, y_label=self.y_label,
            multi_input=self.multi_input,
            batch_size=self.batch_size,
            params=self.params,
            back_perc=self.back_perc if not get_USE_PM() and (get_m() != get_img_width() and get_n() != get_img_weight()) else 100,
            is3dot5DModel=self.is3dot5DModel,
            is4DModel=self.is4DModel,
            inputImgFlag=self.input_img_flag,
            supervised=self.model_info["supervised"],
            patients_folder=self.patients_folder,
            labeled_img_folder=self.labeled_img_folder,
            constants={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                       "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                       "weights": get_weights(), "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(),
                       "USE_PM": get_USE_PM(), "LIST_PMS": get_list_PMS(), "IMAGE_HEIGHT": get_img_weight(),
                       "IMAGE_WIDTH": get_img_width()},
            name=self.name,
            SVO_focus=self.model_info["SVO_focus"],
            loss=self.loss["name"])

        # validation data sequence
        self.val_sequence = sequence_utils.ds_sequence(
            dataframe=self.train_df,
            indices=self.dataset["val"]["indices"],
            sample_weights=self.get_sample_weights("val"),
            x_label=self.x_label,
            y_label=self.y_label,
            multi_input=self.multi_input,
            batch_size=self.batch_size,
            params=self.params,
            back_perc=self.back_perc if not get_USE_PM() and (get_m() != get_img_width() and get_n() != get_img_weight()) else 100,
            is3dot5DModel=self.is3dot5DModel,
            is4DModel=self.is4DModel,
            inputImgFlag=self.input_img_flag,
            supervised=self.model_info["supervised"],
            patients_folder=self.patients_folder,
            labeled_img_folder=self.labeled_img_folder,
            constants={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                       "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                       "weights": get_weights(), "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(),
                       "USE_PM": get_USE_PM(), "LIST_PMS": get_list_PMS(), "IMAGE_HEIGHT": get_img_weight(),
                       "IMAGE_WIDTH": get_img_width()},
            name=self.name,
            flagtype="val",
            loss=self.loss["name"])

    ################################################################################
    # Function to start the train using the sequence as input and the fit_generator function
    def run_train_sequence(self):
        self.train = training.fit_generator(
            model=self.model,
            train_sequence=self.train_sequence,
            val_sequence=self.val_sequence,
            steps_per_epoch=math.ceil((self.train_sequence.__len__()*self.steps_per_epoch_ratio)),
            validation_steps=math.ceil((self.val_sequence.__len__()*self.validation_steps_ratio)),
            epochs=self.epochs,
            callbacklist=self.callbacks,
            initial_epoch=self.initial_epoch,
            use_multiprocessing=self.mp_in_nn
        )

    ################################################################################
    # Check if we need to perform the hybrid solution or not
    def gradual_fine_tuning_solution(self):
        model_name = ""
        # Hybrid solution to fine-tuning the model unfreezing the layers in the VGG-16 architectures
        if "gradual_finetuning_solution" in self.params.keys() and self.params["trainable"] == 0:
            finished_first_half = False
            layer_indexes = []
            if self.params["concatenate_input"]:
                model_name = "model"
                if "nihss" in self.multi_input.keys() and self.multi_input["nihss"] == 1: model_name += "_1"
                elif "age" in self.multi_input.keys() and self.multi_input["age"] == 1: model_name += "_1"
                elif "gender" in self.multi_input.keys() and self.multi_input["gender"] == 1: model_name += "_1"
                layer_indexes.extend([i for i, l in enumerate(self.model.get_layer(model_name).layers) if "concat" in l.name])
            else:
                for pm in get_list_PMS(): layer_indexes.extend([i for i, l in enumerate(self.model.layers) if pm.lower() in l.name])
            layer_indexes = np.sort(layer_indexes)

            # The optimizer (==ADAM) should have a low learning rate
            if self.optimizer_info["name"].lower() != "adam":
                print("The optimizer is not Adam!")
                return

            self.optimizer_info["lr"] = 1e-5
            self.info_callbacks["ModelCheckpoint"]["period"] = 1
            previousEarlyStoppingPatience = self.info_callbacks["EarlyStopping"]["patience"]
            self.info_callbacks["EarlyStopping"]["patience"] = 25

            if self.params["gradual_finetuning_solution"]["type"] == "half":
                # Perform fine tuning twice: first on the bottom half, then on the totality
                # Make the bottom half of the VGG-16 layers trainable
                if self.params["concatenate_input"]:
                    for ind in layer_indexes[len(layer_indexes) // 2:]: self.model.get_layer(model_name).layers[ind].trainable = True
                else:
                    for ind in layer_indexes[len(layer_indexes) // 2:]: self.model.layers[ind].trainable = True
                if is_verbose(): print("Fine-tuning setting: {} layers trainable".format(layer_indexes[len(layer_indexes) // 2:]))
                if self.are_partial_weights_saved():
                    if not self.params["concatenate_input"]: self.model.load_weights(self.partial_weights_path)
                    self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partial_weights_path) + previousEarlyStoppingPatience
                # Compile the model again
                self.compile_model()
                # Get the sample weights
                self.sample_weights = self.get_sample_weights("train")
                # Set the callbacks
                self.set_callbacks(self.sample_weights, "_half")
                # Train the model again
                self.run_train_sequence()
                finished_first_half = False if "only" in self.params["gradual_finetuning_solution"].keys() and self.params["gradual_finetuning_solution"]["only"] == "half" else True

            if self.params["gradual_finetuning_solution"]["type"] == "full" or finished_first_half:
                # Make ALL the VGG-16 layers trainable
                if self.params["concatenate_input"]:
                    for ind in layer_indexes:  self.model.get_layer(model_name).layers[ind].trainable = True
                else:
                    for ind in layer_indexes:  self.model.layers[ind].trainable = True
                if is_verbose(): print("Fine-tuning setting: {} layers trainable".format(layer_indexes))
                if self.are_partial_weights_saved():
                    if not self.params["concatenate_input"]: self.model.load_weights(self.partial_weights_path)
                    self.initial_epoch = general_utils.getEpochFromPartialWeightFilename(self.partial_weights_path) + previousEarlyStoppingPatience
                self.compile_model()  # Compile the model again
                self.sample_weights = self.get_sample_weights("train")  # Get the sample weights
                self.set_callbacks(self.sample_weights, "_full")  # Set the callbacks
                self.run_train_sequence()  # Train the model again

    ################################################################################
    # Get the sample weight from the dataset
    def get_sample_weights(self, flagDataset):
        sample_weights = None
        self.N_BACKGROUND, self.N_BRAIN, self.N_PENUMBRA, self.N_CORE, self.N_TOT = dataset_utils.get_number_of_elem(self.train_df)

        if get_n_classes() ==4:
            if get_m() == get_img_width() and get_n() == get_img_weight():  # and the (M,N) == image dimension
                if self.use_sequence:  # set everything == 1
                    sample_weights = self.train_df.assign(ground_truth=1)
                    sample_weights = sample_weights.ground_truth
                else:  # function that map each getPIXELVALUES()[2] with 150, getPIXELVALUES()[3] with 20 and the rest with 0.1 and sum them
                    f = lambda x: np.sum(np.where(np.array(x) == get_pixel_values()[2], get_weights()[0][3],
                                                  np.where(np.array(x) == get_pixel_values()[3], get_weights()[0][2],
                                                           HOT_ONE_WEIGHTS[0][0])))

                    sample_weights = self.train_df.ground_truth.map(f)
                    sample_weights = sample_weights/(get_m() * get_n())
            else:  # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    get_labels()[0]: self.N_TOT / (get_n_classes() * self.N_BACKGROUND) if self.N_BACKGROUND > 0 else 0,
                    get_labels()[1]: self.N_TOT / (get_n_classes() * self.N_BRAIN) if self.N_BRAIN > 0 else 0,
                    get_labels()[2]: self.N_TOT / (get_n_classes() * self.N_PENUMBRA) if self.N_PENUMBRA > 0 else 0,
                    get_labels()[3]: self.N_TOT / (get_n_classes() * self.N_CORE) if self.N_CORE > 0 else 0,
                })
        elif get_n_classes() ==3:
            if get_m() == get_img_width() and get_n() == get_img_weight():  # and the (M,N) == image dimension
                if self.use_sequence:  # set everything == 1
                    sample_weights = self.train_df.assign(ground_truth=1)
                    sample_weights = sample_weights.ground_truth
                else:  # function that map each getPIXELVALUES()[2] with 150, getPIXELVALUES()[3] with 20 and the rest with 0.1 and sum them
                    f = lambda x: np.sum(np.where(np.array(x)==150,150,np.where(np.array(x)==76,20,0.1)))

                    sample_weights = self.train_df.ground_truth.map(f)
                    sample_weights = sample_weights/(get_m() * get_n())
            else: # see: "ISBI 2019 C-NMC Challenge: Classification in Cancer Cell Imaging" section 4.1 pag 68
                sample_weights = self.train_df.label.map({
                    get_labels()[0]: self.N_TOT / (get_n_classes() * (self.N_BACKGROUND + self.N_BRAIN)) if self.N_BACKGROUND + self.N_BRAIN > 0 else 0,
                    get_labels()[1]: self.N_TOT / (get_n_classes() * self.N_PENUMBRA) if self.N_PENUMBRA > 0 else 0,
                    get_labels()[2]: self.N_TOT / (get_n_classes() * self.N_CORE) if self.N_CORE > 0 else 0,
                })
        elif get_n_classes() ==2:  # we are in a binary class problem
            # f = lambda x : np.sum(np.array(x))
            # sample_weights = self.train_df.ground_truth.map(f)
            sample_weights = self.train_df.label.map({
                get_labels()[0]: self.N_TOT / (get_n_classes() * self.N_BACKGROUND) if self.N_BACKGROUND > 0 else 0,
                get_labels()[1]: self.N_TOT / (get_n_classes() * self.N_CORE) if self.N_CORE > 0 else 0,
            })

        return np.array(sample_weights.values[self.dataset[flagDataset]["indices"]])

    ################################################################################
    # Set the debug set
    def set_debug_ds(self):
        self.val["validation_perc"] = 2
        self.val["number_patients_for_validation"] = 5
        self.val["number_patients_for_testing"] = 0
        self.val["random_validation_selection"] = 0

    ################################################################################
    # Save the trained model and its relative weights
    def save_model_and_weights(self):
        saved_modelname = self.get_saved_model()
        saved_weightname = self.get_saved_weights()

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(saved_modelname, "w") as json_file: json_file.write(model_json)
        self.model.save_weights(saved_weightname)  # serialize weights to HDF5

        if is_verbose():
            general_utils.print_sep("-", 50)
            print("[INFO - Saving] - Saved model and weights to disk!")

    ################################################################################
    # Call the function located in testing for predicting and saved the images
    def predict_and_save_img(self, patientslist, is_already_saved):
        stats = {}
        if is_verbose: print("[INFO] - List of patients to predict: {}".format(patientslist))
        for i, p_id in enumerate(patientslist):  # evaluate the model with the testing patient
            print("[INFO] - Patient {0} -- {1}/{2}".format(p_id, i+1, len(patientslist)))
            if self.model_info["supervised"] and i==0: self.evaluate_model_with_categorics(p_id, is_already_saved, i)
            general_utils.print_sep("+", 50)
            print("[INFO] - Executing function: predictAndSaveImages for patient {}".format(p_id))
            testing.predict_and_save_img(self, p_id)

        return stats

    ################################################################################
    # Test the model with the selected patient (if the number of patient to test is > 0)
    def evaluate_model_with_categorics(self, p_id, is_already_saved, i):
        if is_verbose():
            general_utils.print_sep("+", 50)
            print("[INFO] - Evaluating the model for patient {}".format(p_id))

        self.testing_score.append(testing.evaluate_model(self, p_id, is_already_saved, i))

    ################################################################################
    # set the flag for single/multi PROCESSING
    def set_processing_env(self, mp):
        self.mp = mp
        self.mp_in_nn = False

    ################################################################################
    # return the saved model or weight (based on the suffix)
    def getSavedInformation(self, path, other_info="", suffix=""):
        # mJ-Net_DA_ADAM_4_16x16.json <-- example weights name
        # mJ-Net_DA_ADAM_4_16x16.h5 <-- example model name
        path = general_utils.get_dir_path(path) + self.get_nn_id() + other_info + general_utils.get_suffix()
        return path+suffix

    ################################################################################
    # return the saved model
    def get_saved_model(self):
        return self.getSavedInformation(path=self.saved_model_folder, suffix=".json")

    ################################################################################
    # return the saved weight
    def get_saved_weights(self):
        return self.getSavedInformation(path=self.saved_model_folder, suffix=".h5")

    ################################################################################
    # return NeuralNetwork ID
    def get_nn_id(self):
        # CAREFUL WITH THIS !!
        # needs to override the model id to use a different model to test various patients
        if self.OVERRIDE_MODELS_ID_PATH: ret_id = self.OVERRIDE_MODELS_ID_PATH
        else:
            ret_id = self.name
            if self.model_info["data_augmentation"]: ret_id += "_DA"
            ret_id += ("_" + self.optimizer_info["name"].upper())

            ret_id += ("_VAL" + str(self.val["validation_perc"]))
            if self.val["random_validation_selection"]: ret_id += "_RANDOM"

            if is_TO_CATEG(): ret_id += "_SOFTMAX"  # differentiate between softmax and sigmoid last activation layer

            # if there is cross validation, add the SPLIT ID to differentiate the models
            if self.cross_validation["use"]: ret_id += ("_" + self.model_split)

        return ret_id
