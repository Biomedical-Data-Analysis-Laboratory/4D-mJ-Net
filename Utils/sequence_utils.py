import math, cv2, glob, time
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

import constants
from Utils import general_utils

################################################################################
# https://faroit.com/keras-docs/2.1.3/models/sequential/#fit_generator
class trainValSequence(Sequence):
    def __init__(self, dataframe, indices, sample_weights, x_label, y_label, batch_size):
        self.dataframe = dataframe
        self.indices = indices
        self.x_label = x_label
        self.y_label = y_label
        self.batch_size = batch_size
        self.sample_weights = sample_weights

        self.columns = self.dataframe.columns
        self.dataframe = pd.DataFrame(self.dataframe.values[self.indices], columns=self.columns)

    # Every Sequence must implement the __getitem__ and the __len__ methods
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size
        index_pd = list(range(start, end))

        index_pd_DA = {
            "0": set(),
            "1": set(),
            "2": set(),
            "3": set(),
            "4": set(),
            "5": set()}

        pixels_filenames = (self.dataframe[self.x_label][start:end], self.dataframe["x_y"][start:end], self.dataframe["data_aug_idx"][start:end])
        batch_y = (self.dataframe[self.y_label][start:end], self.dataframe["x_y"][start:end])
        weights = self.sample_weights[start:end]


        if self.x_label=="pixels": # path to the forlder containing the NUMBER_OF_IMAGE_PER_SECTION   time point images
            X = np.empty((len(self.dataframe[self.x_label][start:end]),constants.getM(),constants.getN(),constants.NUMBER_OF_IMAGE_PER_SECTION, 1)) # empty array for the pixels
            for i,folder in enumerate(pixels_filenames[0]):
                for timeIndex,filename in enumerate(np.sort(glob.glob(folder+"*"+constants.SUFFIX_IMG))):
                    # for ISLES2018 (to change in the future) --> if the number of timepoint per slice is > constants.NUMBER_OF_IMAGE_PER_SECTION
                    # break the loop and take only the first constants.NUMBER_OF_IMAGE_PER_SECTION
                    # TODO: change it into INTERPOLATION to have the same number of images and use the entire timepoints
                    if timeIndex>=constants.NUMBER_OF_IMAGE_PER_SECTION: break

                    tmp = general_utils.getSlicingWindow(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),
                            pixels_filenames[1][index_pd[i]][0], pixels_filenames[1][index_pd[i]][1], constants.getM(), constants.getN())

                    if pixels_filenames[2][index_pd[i]]==1: tmp = np.rot90(tmp) # rotate 90 degree counterclockwise
                    elif pixels_filenames[2][index_pd[i]]==2: tmp = np.rot90(tmp,2) # rotate 180 degree counterclockwise
                    elif pixels_filenames[2][index_pd[i]]==3: tmp = np.rot90(tmp,3) # rotate 270 degree counterclockwise
                    elif pixels_filenames[2][index_pd[i]]==4: tmp = np.flipud(tmp) # rotate 270 degree counterclockwise
                    elif pixels_filenames[2][index_pd[i]]==5: tmp = np.fliplr(tmp) # flip the matrix left/right
                    # add the index into the correct set
                    index_pd_DA[str(pixels_filenames[2][index_pd[i]])].add(i)
                    # reshape it for the correct input in the model
                    X[i,:,:,timeIndex,:] = tmp.reshape(tmp.shape+(1,))

            if type(X) is not np.ndarray: X = np.array(X)

        if self.y_label=="ground_truth": # path to the ground truth image
            Y = np.empty((len(self.dataframe[self.y_label][start:end]),constants.getM(),constants.getN()))
            for aug_idx in index_pd_DA.keys():
                if aug_idx=="0":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN())
                elif aug_idx=="1":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = np.rot90(general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN()))
                elif aug_idx=="2":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = np.rot90(general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN()),2)
                elif aug_idx=="3":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = np.rot90(general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN()),3)
                elif aug_idx=="4":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = np.flipud(general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN()))
                elif aug_idx=="5":
                    for i in index_pd_DA[aug_idx]: Y[i,:,:] = np.fliplr(general_utils.getSlicingWindow(cv2.imread(batch_y[0][index_pd[i]], cv2.IMREAD_GRAYSCALE), batch_y[1][index_pd[i]][0], batch_y[1][index_pd[i]][1], constants.getM(), constants.getN()))

        # a tuple (inputs, targets, sample_weights). All arrays should contain the same number of samples.
        return X, Y, weights
