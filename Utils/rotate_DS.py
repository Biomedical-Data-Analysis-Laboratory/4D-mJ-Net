#!/usr/bin/env python
# coding: utf-8

################################################################################
# ### Import libraries
import cv2, time, glob, os, operator, random, math
import numpy as np
from scipy import ndimage

################################################################################
TEST_SET = ["01_001","01_007","01_013","01_019","01_025","01_031",
    "01_037","01_044","01_049","01_053","01_061","01_067","01_074",
    "02_001","02_007","02_013","02_019","02_025","02_031","02_036",
    "02_043","02_050","02_055","02_062","03_003","03_010","03_014",
    "01_057","01_059","01_066","01_068","01_071","01_073"] # ignore them
IMAGE_PREFIX = "CTP_"
ROOT_PATH = "/home/prosjekt/PerfusionCT/StrokeSUS/"
DS_NAME = "FINAL_TIFF_HU_v2/"
ORIGINAL_FOLDER = ROOT_PATH + DS_NAME
SAVE_REGISTERED_FOLDER = ROOT_PATH + "MIRRORED_" + DS_NAME

################################################################################
def readAndMirrorImages():
    patientFolders = glob.glob(ORIGINAL_FOLDER + "*/")

    for numFold, patientFolder in enumerate(patientFolders):  # for each patient
        relativePatientPath = patientFolder.replace(ORIGINAL_FOLDER, '')
        if relativePatientPath[:-1].replace(IMAGE_PREFIX,"") not in TEST_SET:
            print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold + 1, len(patientFolders), relativePatientPath))
            if not os.path.isdir(SAVE_REGISTERED_FOLDER+relativePatientPath): os.mkdir(SAVE_REGISTERED_FOLDER+relativePatientPath)

            for subfold in glob.glob(patientFolder + "*/"):
                slicefold = subfold.replace(patientFolder,'')
                if not os.path.isdir(SAVE_REGISTERED_FOLDER+relativePatientPath+slicefold): os.mkdir(SAVE_REGISTERED_FOLDER+relativePatientPath+slicefold)

                for image_name in glob.glob(subfold + "*"):
                    image_idx = image_name.replace(subfold,'')
                    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                    mirror_img = np.fliplr(img)
                    cv2.imwrite(SAVE_REGISTERED_FOLDER+relativePatientPath+slicefold+image_idx, mirror_img)

################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    start = time.time()
    if not os.path.isdir(SAVE_REGISTERED_FOLDER): os.mkdir(SAVE_REGISTERED_FOLDER)
    print("Mirror the dataset...")
    readAndMirrorImages()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))