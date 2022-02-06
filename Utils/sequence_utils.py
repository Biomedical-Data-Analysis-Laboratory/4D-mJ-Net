import warnings

import math, cv2, platform
from typing import Set, Dict, Any

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from Model.constants import *
from Utils import general_utils, dataset_utils, model_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class ds_sequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label, multi_input, batch_size, params, back_perc,
                 is3dot5DModel, is4DModel, inputImgFlag, supervised, patients_folder, labeled_img_folder, constants,
                 name, SVO_focus=False, flagtype="train", loss=None):
        self.indices = indices
        self.dataframe = dataframe.iloc[self.indices]
        self.name = name
        self.sample_weights = sample_weights
        self.x_label = x_label
        self.y_label = y_label
        self.multi_input = multi_input
        self.batch_size = batch_size
        self.params = params
        self.back_perc = back_perc
        self.flag_type = flagtype
        self.loss = loss
        self.is3dot5DModel = is3dot5DModel
        self.is4DModel = is4DModel
        self.SVO_focus = SVO_focus
        self.inputImgFlag = inputImgFlag  # only works when the input are the PMs (concatenate)
        self.supervised = supervised
        self.patients_folder = patients_folder
        self.labeled_img_folder = labeled_img_folder
        self.constants = constants

        if self.flag_type != "test":
            # get ALL the rows with label != from background
            self.df_noback = self.dataframe.loc[self.dataframe.label != get_labels()[0]]
            # also, get a back_perc of rows with label == background
            self.df_back = self.dataframe.loc[self.dataframe.label == get_labels()[0]]
            if self.back_perc < 100: self.df_back = self.df_back[:int((len(self.df_back) / 100) * self.back_perc)]
            # combine the two dataframes
            self.dataframe = pd.concat([self.df_noback, self.df_back], sort=False)

        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}
        self.index_batch = None

        self.n_slices = 0 if "n_slices" not in self.params.keys() else self.params["n_slices"]

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1)  # shuffle the dataframe rows at the end of an epoch

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size

        current_batch = self.dataframe[start:end]
        self.index_batch = current_batch.index

        # empty initialization
        X = np.empty((len(current_batch), self.constants["NUMBER_OF_IMAGE_PER_SECTION"], self.constants["M"], self.constants["N"], 1))
        if self.constants["TIME_LAST"]: X = np.empty((len(current_batch), self.constants["M"], self.constants["N"], self.constants["NUMBER_OF_IMAGE_PER_SECTION"], 1))
        Y = []
        weights = np.empty((len(current_batch),))

        if self.constants["USE_PM"]: X = np.empty((len(current_batch), self.constants["M"], self.constants["N"]))

        # reset the index for the data augmentation
        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

        # path to the folder containing the getNUMBER_OF_IMAGE_PER_SECTION() time point images
        X, Y, weights = self.get_XY(X, Y, weights, current_batch)

        return X, Y, weights

    ################################################################################
    # return the X set and the relative weights based on the pixels column
    def get_XY(self, X, Y, weights, current_batch):
        for index, (row_index, row) in enumerate(current_batch.iterrows()):
            # add the index into the correct set
            self.index_pd_DA[str(row["data_aug_idx"])].add(row_index)
            X = model_utils.get_correct_X_for_input_model(self, row[self.x_label], row, batch_idx=index,
                                                          batch_len=len(current_batch), X=X, train=True)
            if self.y_label=="ground_truth": Y, weights = self.get_Y(Y, row, index, weights)

        return X, np.array(Y), weights

    ################################################################################
    # Return the Y set and the weights
    def get_Y(self, Y, row, index, weights):
        aug_idx = str(row["data_aug_idx"])  # index if there's augmentation
        filename = row[self.y_label]
        if platform.system()=="Windows": filename = filename.replace(filename[:filename.rfind("/",0,len(filename)-7)], self.labeled_img_folder)
        if not isinstance(filename,str): print(filename)

        coord = row["x_y"]  # coordinates of the slice window
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        assert img is not None, "The image {} is None".format(filename)
        img = general_utils.get_slice_window(img, coord[0], coord[1], self.constants, is_gt=True)

        # remove the brain from the image ==> it becomes background
        if self.constants["N_CLASSES"]<=3: img[img == 85] = self.constants["PIXELVALUES"][0]
        # remove the penumbra ==> it becomes core
        if self.constants["N_CLASSES"]==2: img[img == 170] = self.constants["PIXELVALUES"][1]

        # Override the weights based on the pixel values
        if self.constants["N_CLASSES"]>2:
            core_idx, penumbra_idx = 3, 2
            if self.constants["N_CLASSES"] == 3: core_idx, penumbra_idx = 2, 1
            core_value, core_weight = self.constants["PIXELVALUES"][core_idx], self.constants["weights"][0][core_idx]
            penumbra_value, penumbra_weight = self.constants["PIXELVALUES"][penumbra_idx], self.constants["weights"][0][penumbra_idx]

            # focus on the SVO core only during training (only for SUS2020 dataset)!
            if self.SVO_focus and row["severity"]=="02":
                core_weight *= 6
                penumbra_weight *= 6

            # sum the pixel value for the image with the corresponding "weight" for class
            sumpixweight = lambda x: np.sum(np.where(np.array(x) == core_value, core_weight,
                                                     np.where(np.array(x) == penumbra_value, penumbra_weight, self.constants["weights"][0][0])))
            weights[index] = sumpixweight(img) / (self.constants["M"] * self.constants["N"])
        elif self.constants["N_CLASSES"] == 2:
            core_value, core_weight = self.constants["PIXELVALUES"][1], self.constants["weights"][0][1]
            sumpixweight = lambda x: np.sum(np.where(np.array(x) == core_value, core_weight, self.constants["weights"][0][0]))
            weights[index] = sumpixweight(img) / (self.constants["M"] * self.constants["N"])

        # convert the label in [0, 1] values,
        # for to_categ the division happens inside dataset_utils.getSingleLabelFromIndexCateg
        if not self.constants["TO_CATEG"]: img = np.divide(img, 255)
        # Perform data augmentation on the ground truth
        img = general_utils.perform_DA_on_img(img, int(aug_idx))
        if self.constants["TO_CATEG"] and not self.loss == "sparse_categorical_crossentropy":
            # Convert the image to categorical if needed
            img = dataset_utils.get_single_label_from_idx_categ(img, self.constants["N_CLASSES"])

        Y.append(img)

        return Y, weights
