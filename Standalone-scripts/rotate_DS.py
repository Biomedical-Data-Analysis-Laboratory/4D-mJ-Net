#!/usr/bin/env python
# coding: utf-8

################################################################################
# ### Import libraries
import cv2, time, glob, os, operator, random, math, argparse
import numpy as np
from scipy import ndimage

################################################################################
IMAGE_PREFIX = "CTP_"

################################################################################
def readAndMirrorImages(ORIGINAL_FOLDER,MIRRORED_REGISTERED_FOLDER,MIRRORED_GT_FOLDER):
    patientFolders = glob.glob(ORIGINAL_FOLDER + "*/")

    for numFold, patientFolder in enumerate(patientFolders):  # for each patient
        relativePatientPath = patientFolder.replace(ORIGINAL_FOLDER, '')

        print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold + 1, len(patientFolders), relativePatientPath))

        readAndMirrorGT(relativePatientPath,MIRRORED_GT_FOLDER)

        if not os.path.isdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath): os.mkdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath)
        else:
            print("Directory {} already exist, continue...".format(MIRRORED_REGISTERED_FOLDER + relativePatientPath))
            continue

        for subfold in glob.glob(patientFolder + "*/"):
            slicefold = subfold.replace(patientFolder,'')
            if not os.path.isdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold): os.mkdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold)

            for image_name in glob.glob(subfold + "*"):
                image_idx = image_name.replace(subfold,'')
                img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                mirror_img = np.fliplr(img)
                cv2.imwrite(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold + image_idx, mirror_img)


################################################################################
def readAndMirrorGT(relativePatientPath,MIRRORED_GT_FOLDER):
    if not os.path.isdir(MIRRORED_GT_FOLDER + relativePatientPath):
        os.mkdir(MIRRORED_GT_FOLDER + relativePatientPath)
        for image_name in glob.glob(GT_FOLDER+relativePatientPath + "*"):
            image_idx = image_name.replace(GT_FOLDER+relativePatientPath, '')
            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            mirror_img = np.fliplr(img)
            cv2.imwrite(MIRRORED_GT_FOLDER + relativePatientPath + image_idx, mirror_img)


################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Set the root folder")
    parser.add_argument("ds_name", help="Set the dataset folder name")
    parser.add_argument("gt_name", help="Set the ground truth folder name")
    args = parser.parse_args()

    ROOT_PATH = args.root
    DS_NAME = args.ds_name
    GT_NAME = args.gt_name
    ORIGINAL_FOLDER = ROOT_PATH + DS_NAME
    MIRRORED_REGISTERED_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + DS_NAME
    GT_FOLDER = ROOT_PATH + GT_NAME
    MIRRORED_GT_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + GT_NAME

    start = time.time()
    if not os.path.isdir(MIRRORED_REGISTERED_FOLDER): os.mkdir(MIRRORED_REGISTERED_FOLDER)
    if not os.path.isdir(MIRRORED_GT_FOLDER): os.mkdir(MIRRORED_GT_FOLDER)

    print("Mirror the dataset...")
    readAndMirrorImages(ORIGINAL_FOLDER,MIRRORED_REGISTERED_FOLDER,MIRRORED_GT_FOLDER)
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))