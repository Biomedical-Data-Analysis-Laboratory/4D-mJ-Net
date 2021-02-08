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
def readAndMirrorImages():
    patientFolders = glob.glob(ORIGINAL_FOLDER + "*/")

    for numFold, patientFolder in enumerate(patientFolders):  # for each patient
        relativePatientPath = patientFolder.replace(ORIGINAL_FOLDER, '')

        print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold + 1, len(patientFolders), relativePatientPath))

        readAndMirrorGT(relativePatientPath)

        readAndMirrorPM(relativePatientPath)

        readAndMirror4DCTP(relativePatientPath,patientFolder)

        readAndMirrorMask(relativePatientPath)

################################################################################
def readAndMirror4DCTP(relativePatientPath,patientFolder):
    if not os.path.isdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath): os.mkdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath)
    else:
        print("Directory {} already exist, continue...".format(MIRRORED_REGISTERED_FOLDER + relativePatientPath))
        return

    for subfold in glob.glob(patientFolder + "*/"):
        slicefold = subfold.replace(patientFolder,'')
        if not os.path.isdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold): os.mkdir(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold)

        for image_name in glob.glob(subfold + "*"):
            image_idx = image_name.replace(subfold,'')
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            mirror_img = np.fliplr(img)
            cv2.imwrite(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold + image_idx, mirror_img)


################################################################################
def readAndMirrorPM(relativePatientPath):
    if not os.path.isdir(MIRRORED_PM_FOLDER + relativePatientPath): os.mkdir(MIRRORED_PM_FOLDER + relativePatientPath)
    else:
        print("PMs for {} already exists, continue...".format(MIRRORED_PM_FOLDER + relativePatientPath))
        return

    for dayfolder in glob.glob(PM_FOLDER + relativePatientPath + "*/"):
        # if the folder contains the correct number of subfolders
        if len(glob.glob(dayfolder + "*/")) >= 7:
            pmlist = ["CBF", "CBV", "TTP", "TMAX", "MIP"]
            day_f = dayfolder.replace(PM_FOLDER+relativePatientPath, '')
            if not os.path.isdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f): os.mkdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f)
            for listpms in glob.glob(dayfolder + "*/"):
                for pm in pmlist:
                    if pm in listpms:
                        if not os.path.isdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f+pm+"/"): os.mkdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f+pm+"/")
                        #print(glob.glob(listpms+"*"))
                        for image_name in glob.glob(listpms+"*"):
                            image_idx = image_name.replace(PM_FOLDER+relativePatientPath+day_f+pm+"/", '')
                            img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                            mirror_img = np.fliplr(img)

                            mirror_img[:,435:] = img[:,435:]  # add the colorbar on the right in the mirrored image
                            mirror_img[:,:512-435] = 0  # remove the colorbar on the left in the mirrored image

                            cv2.imwrite(MIRRORED_PM_FOLDER+relativePatientPath+day_f+pm+"/"+image_idx, mirror_img)


################################################################################
def readAndMirrorGT(relativePatientPath):
    if not os.path.isdir(MIRRORED_GT_FOLDER + relativePatientPath): os.mkdir(MIRRORED_GT_FOLDER + relativePatientPath)
    else:
        print("GT for {} already exists, continue...".format(MIRRORED_GT_FOLDER + relativePatientPath))
        return

    for image_name in glob.glob(GT_FOLDER+relativePatientPath + "*"):
        image_idx = image_name.replace(GT_FOLDER+relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        mirror_img = np.fliplr(img)
        cv2.imwrite(MIRRORED_GT_FOLDER + relativePatientPath + image_idx, mirror_img)


def readAndMirrorMask(relativePatientPath):
    if not os.path.isdir(MIRRORED_MASK_FOLDER + relativePatientPath): os.mkdir(MIRRORED_MASK_FOLDER + relativePatientPath)
    else:
        print("MAsk for {} already exists, continue...".format(MIRRORED_MASK_FOLDER + relativePatientPath))
        return

    for image_name in glob.glob(MASK_FOLDER + relativePatientPath + "*"):
        image_idx = image_name.replace(MASK_FOLDER + relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        mirror_img = np.fliplr(img)
        cv2.imwrite(MIRRORED_MASK_FOLDER + relativePatientPath + image_idx, mirror_img)


################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Set the root folder")
    parser.add_argument("ds_name", help="Set the dataset folder name")
    parser.add_argument("pm_name", help="Set the pms folder name")
    parser.add_argument("gt_name", help="Set the ground truth folder name")
    parser.add_argument("mask_name", help="Set the mask folder name")
    args = parser.parse_args()

    ROOT_PATH = args.root
    DS_NAME = args.ds_name
    PM_NAME = args.pm_name
    GT_NAME = args.gt_name
    MASK_NAME = args.mask_name
    ORIGINAL_FOLDER = ROOT_PATH + DS_NAME
    MIRRORED_REGISTERED_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + DS_NAME
    PM_FOLDER = ROOT_PATH + PM_NAME
    MIRRORED_PM_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + PM_NAME
    GT_FOLDER = ROOT_PATH + GT_NAME
    MIRRORED_GT_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + GT_NAME
    MASK_FOLDER = ROOT_PATH + MASK_NAME
    MIRRORED_MASK_FOLDER = ROOT_PATH + "MIRRORED/MIRRORED_" + MASK_NAME

    start = time.time()
    if not os.path.isdir(MIRRORED_REGISTERED_FOLDER): os.mkdir(MIRRORED_REGISTERED_FOLDER)
    if not os.path.isdir(MIRRORED_PM_FOLDER): os.mkdir(MIRRORED_PM_FOLDER)
    if not os.path.isdir(MIRRORED_GT_FOLDER): os.mkdir(MIRRORED_GT_FOLDER)
    if not os.path.isdir(MIRRORED_MASK_FOLDER): os.mkdir(MIRRORED_MASK_FOLDER)

    print("Mirror the dataset...")
    readAndMirrorImages()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))