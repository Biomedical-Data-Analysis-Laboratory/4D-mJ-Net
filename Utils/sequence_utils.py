import math, cv2, glob, time, os
from typing import Set, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import constants
from Utils import general_utils, dataset_utils


################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class datasetSequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label,
                 to_categ, batch_size, back_perc=2, istest=False, loss=None):
        self.indices = indices
        self.dataframe = dataframe.iloc[self.indices]

        self.sample_weights = sample_weights
        self.x_label = x_label
        self.y_label = y_label
        self.batch_size = batch_size
        self.to_categ = to_categ
        self.back_perc = back_perc
        self.loss = loss

        if not istest:
            # get ALL the rows with label != from background
            self.dataframenoback = self.dataframe.loc[self.dataframe.label != constants.LABELS[0]]
            # also, get a back_perc of rows with label == background
            self.dataframeback = self.dataframe.loc[self.dataframe.label == constants.LABELS[0]]
            if self.back_perc < 100: self.dataframeback = self.dataframeback[:int((len(self.dataframeback)/100)*self.back_perc)]
            # combine the two dataframes
            self.dataframe = pd.concat([self.dataframenoback, self.dataframeback], sort=False)

        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size

        batch_index_list = list(range(start,end))
        current_batch = self.dataframe[start:end]

        # empty initialization
        X = np.empty((len(current_batch),constants.getM(),constants.getN(),constants.NUMBER_OF_IMAGE_PER_SECTION,1))
        weights = np.empty((len(current_batch),))

        if constants.getUSE_PM(): X = np.empty((len(current_batch), constants.getM(), constants.getN()))

        # reset the index for the data augmentation
        self.index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

        # path to the folder containing the NUMBER_OF_IMAGE_PER_SECTION time point images
        if self.x_label=="pixels":  X, weights = self.getX_pixels(X,current_batch,weights,batch_index_list)
        elif self.x_label==["CBF","CBV","TTP","TMAX"]: X, weights = self.getX_PM(weights,current_batch,batch_index_list)

        # path to the ground truth image
        if self.y_label=="ground_truth": Y, weights = self.getY(current_batch,weights)

        # Check if any value is NaN
        if np.isnan(np.unique(Y)).any(): print([gt for gt in current_batch.ground_truth])

        return X, Y, weights

    ################################################################################
    # return the X set and the relative weights based on the pixels column
    def getX_pixels(self, X, current_batch, weights, batch_index_list):
        for index, (row_index, row) in enumerate(current_batch.iterrows()):
            weights[index] = self.sample_weights[batch_index_list[index]]
            folder = row[self.x_label]
            coord = row["x_y"]
            data_aug_idx = row["data_aug_idx"]
            # add the index into the correct set
            self.index_pd_DA[str(data_aug_idx)].add(row_index)

            for timeIndex, filename in enumerate(np.sort(glob.glob(folder + "*" + constants.SUFFIX_IMG))):
                # TODO: for ISLES2018 (to change in the future) --> if the number of time-points per slice
                #  is > constants.NUMBER_OF_IMAGE_PER_SECTION
                if timeIndex >= constants.NUMBER_OF_IMAGE_PER_SECTION: break

                sliceW = general_utils.getSlicingWindow(cv2.imread(filename,cv2.IMREAD_GRAYSCALE),coord[0],coord[1])
                sliceW = general_utils.performDataAugmentationOnTheImage(sliceW, data_aug_idx)

                # reshape it for the correct input in the model
                X[index, :, :, timeIndex, :] = sliceW.reshape(sliceW.shape + (1,))

        return X, weights

    ################################################################################
    # Return the X set and relative weights based on the parametric maps columns
    def getX_PM(self, weights, current_batch, batch_index_list):
        pms = dict()

        for index, (row_index, row) in enumerate(current_batch.iterrows()):
            weights[index] = self.sample_weights[batch_index_list[index]]
            coord = row["x_y"]
            data_aug_idx = row["data_aug_idx"]
            # add the index into the correct set
            self.index_pd_DA[str(data_aug_idx)].add(row_index)

            for pm in self.x_label:
                if pm not in pms.keys(): pms[pm] = []
                pm_filename = row[pm] + general_utils.getStringFromIndex(row["sliceIndex"]) + ".png"

                totimg = img_to_array(load_img(pm_filename))
                if totimg is not None:
                    img = general_utils.getSlicingWindow(totimg, coord[0], coord[1], removeColorBar=True)
                    img = general_utils.performDataAugmentationOnTheImage(img, data_aug_idx)
                    pms[pm].append(img)

        X = [np.array(pms["CBF"]), np.array(pms["CBV"]), np.array(pms["TTP"]), np.array(pms["TMAX"])]

        return X, weights

    ################################################################################
    # Return the Y set and the weights
    def getY(self, current_batch, weights):
        Y = []
        for aug_idx in self.index_pd_DA.keys():
            for index, row_index in enumerate(self.index_pd_DA[aug_idx]):
                filename = current_batch.loc[row_index][self.y_label]
                if not isinstance(filename, str):
                    print(filename)
                    filename = current_batch.loc[row_index][self.y_label][0]

                coord = current_batch.loc[row_index]["x_y"]

                img = img_to_array(load_img(filename, color_mode="grayscale"))
                img = img.reshape(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)

                if ".tiff" in filename: img = np.rint(img/256)

                img = general_utils.getSlicingWindow(img, coord[0], coord[1], isgt=True)
                if not self.to_categ: img /= 256  # convert the label in [0, 1] values

                if aug_idx=="0": img = img if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(img)
                elif aug_idx=="1": img = np.rot90(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img))
                elif aug_idx=="2": img = np.rot90(img, 2) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 2))
                elif aug_idx=="3": img = np.rot90(img, 3) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(img, 3))
                elif aug_idx=="4": img = np.flipud(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.flipud(img))
                elif aug_idx=="5": img = np.fliplr(img) if not self.to_categ or self.loss=="sparse_categorical_crossentropy" else dataset_utils.getSingleLabelFromIndexCateg(np.fliplr(img))

                Y.append(img)

        return np.array(Y), weights
