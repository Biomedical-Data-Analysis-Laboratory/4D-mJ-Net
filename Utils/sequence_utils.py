import math, cv2, glob, time, os
from typing import Set, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import constants
from Utils import general_utils, dataset_utils

################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class datasetSequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label, to_categ, batch_size, back_perc=2):
        self.indices = indices
        self.dataframe = dataframe.iloc[self.indices]

        self.sample_weights = sample_weights
        self.x_label = x_label
        self.y_label = y_label
        self.batch_size = batch_size
        self.to_categ = to_categ
        self.back_perc = back_perc

        # get ALL the rows with label != from background
        self.dataframenoback = self.dataframe.loc[self.dataframe.label != constants.LABELS[0]]
        # also, get a back_perc of rows with label == background
        self.dataframeback = self.dataframe.loc[self.dataframe.label == constants.LABELS[0]]
        self.dataframeback = self.dataframeback[:int((len(self.dataframeback)/100)*self.back_perc)]
        # combine the two dataframes
        self.dataframe = pd.concat([self.dataframenoback, self.dataframeback], sort=False)

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size

        batch_index_list = list(range(start,end))
        current_batch = self.dataframe[start:end]
        weights = np.empty((len(current_batch),))
        # create a list containing the VALID values (inside the valid_indices list) from start to end
        # index_pd = list(range(start, end))

        index_pd_DA: Dict[str, Set[Any]] = {"0": set(),"1": set(),"2": set(),"3": set(),"4": set(),"5": set()}

        if self.x_label=="pixels":  # path to the folder containing the NUMBER_OF_IMAGE_PER_SECTION   time point images
            X = np.empty((len(current_batch),constants.getM(),constants.getN(),
                constants.NUMBER_OF_IMAGE_PER_SECTION, 1))  # empty array for the pixels

            for index,(row_index,row) in enumerate(current_batch.iterrows()):
                weights[index] = self.sample_weights[batch_index_list[index]]
                folder = row[self.x_label]
                coord = row["x_y"]
                data_aug_idx = row["data_aug_idx"]
                # add the index into the correct set
                index_pd_DA[str(data_aug_idx)].add(row_index)

                for timeIndex,filename in enumerate(np.sort(glob.glob(folder+"*"+constants.SUFFIX_IMG))):
                    # TODO: for ISLES2018 (to change in the future) --> if the number of timepoint per slice is > constants.NUMBER_OF_IMAGE_PER_SECTION
                    if timeIndex>=constants.NUMBER_OF_IMAGE_PER_SECTION: break

                    sliceWindow = general_utils.getSlicingWindow(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),
                                                                 coord[0], coord[1], constants.getM(), constants.getN())

                    if data_aug_idx==1: sliceWindow = np.rot90(sliceWindow)  # rotate 90 degree counterclockwise
                    elif data_aug_idx==2: sliceWindow = np.rot90(sliceWindow,2)  # rotate 180 degree counterclockwise
                    elif data_aug_idx==3: sliceWindow = np.rot90(sliceWindow,3)  # rotate 270 degree counterclockwise
                    elif data_aug_idx==4: sliceWindow = np.flipud(sliceWindow)  # rotate 270 degree counterclockwise
                    elif data_aug_idx==5: sliceWindow = np.fliplr(sliceWindow)  # flip the matrix left/right
                    # reshape it for the correct input in the model
                    X[index,:,:,timeIndex,:] = sliceWindow.reshape(sliceWindow.shape+(1,))

            # if type(X) is not np.ndarray: X = np.array(X)

        if self.y_label=="ground_truth":  # path to the ground truth image
            Y = np.empty((len(current_batch),constants.getM(),constants.getN())) if not self.to_categ else np.empty((len(current_batch),constants.getM(),constants.getN(), constants.N_CLASSES))
            for aug_idx in index_pd_DA.keys():
                for index,row_index in enumerate(index_pd_DA[aug_idx]):
                    filename = current_batch.loc[row_index][self.y_label]
                    if not isinstance(filename,str) or not os.path.isfile(filename): print(filename)
                    coord = current_batch.loc[row_index]["x_y"]

                    try:
                        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    except SystemError as e:
                        print("\n filename: {}".format(filename))
                        continue

                    sliceWindow = general_utils.getSlicingWindow(img, coord[0], coord[1], constants.getM(), constants.getN())
                    if aug_idx=="0": Y[index,:,:] = sliceWindow if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(sliceWindow)
                    elif aug_idx=="1": Y[index,:,:] = np.rot90(sliceWindow) if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(sliceWindow))
                    elif aug_idx=="2": Y[index,:,:] = np.rot90(sliceWindow,2) if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(sliceWindow,2))
                    elif aug_idx=="3": Y[index,:,:] = np.rot90(sliceWindow,3) if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(np.rot90(sliceWindow,3))
                    elif aug_idx=="4": Y[index,:,:] = np.flipud(sliceWindow) if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(np.flipud(sliceWindow))
                    elif aug_idx=="5": Y[index,:,:] = np.fliplr(sliceWindow) if not self.to_categ else dataset_utils.getSingleLabelFromIndexCateg(np.fliplr(sliceWindow))

                    # if there are any NaN elements, don't return the batch
                    if np.isnan(Y).any():
                        where = list(map(list,np.argwhere(np.isnan(Y))))
                        for w in where:
                            Y[w[0],w[1],w[2]] = constants.PIXELVALUES[0]
                            weights[w[0]] = 1
        # a tuple (inputs, targets, sample_weights). All arrays should contain the same number of samples.
        if X.shape[0] != Y.shape[0] or X.shape[0] != weights.shape[0] or Y.shape[0] != weights.shape[0]:
            print("different shape: {0},{1},{2}".format(X.shape[0],Y.shape[0],weights.shape[0]))
            return

        return X, Y, weights
