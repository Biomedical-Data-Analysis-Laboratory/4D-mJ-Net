#!/usr/bin/env python
# coding: utf-8

################################################################################
# ### Import libraries
import cv2, time, glob, os, operator, random, math, argparse
import numpy as np
from scipy import ndimage
import statistics

################################################################################
IMAGE_PREFIX = "PA"


################################################################################
def verticalAlignment():
    patientFolders = glob.glob(ORIGINAL_FOLDER + "*/")

    for numFold, patientFolder in enumerate(patientFolders):  # for each patient
        relativePatientPath = patientFolder.replace(ORIGINAL_FOLDER, '')
        print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold + 1, len(patientFolders), relativePatientPath))

        readAndAlignGT(relativePatientPath)

        # readAndAlignPM(relativePatientPath)
        #
        # readAndAlign4DCTP(relativePatientPath, patientFolder)
        #
        # if MASK_NAME != "": readAndAlignMask(relativePatientPath)


################################################################################
def readAndAlignGT(relativePatientPath):
    if not os.path.isdir(ALIGNED_GT_FOLDER + relativePatientPath): os.mkdir(ALIGNED_GT_FOLDER + relativePatientPath)
    # else:
    #     print("GT for {} already exists, continue...".format(ALIGNED_GT_FOLDER + relativePatientPath))
    #     return
    angles = []
    for image_name in np.sort(glob.glob(GT_FOLDER+relativePatientPath + "*")):
        print(image_name)
        image_idx = image_name.replace(GT_FOLDER+relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        center = (img.shape[0]/2,img.shape[1]/2)

        _, bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours)>1: print(image_idx)

        area = cv2.contourArea(contours[0])
        print(area)

        [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
        # Now find two extreme points on the line to draw line
        lefty = int((-x * vy / vx) + y)
        righty = int(((img.shape[1] - x) * vy / vx) + y)

        print(lefty, righty, vx, vy, x, y)

        cv2.line(img, (img.shape[1] - 1, righty), (0, lefty), 255, 2)
        yy = righty
        xx = img.shape[1]-1
        angle = np.arctan2(yy,xx) * (180/np.pi)
        print(np.abs(angle))
        angles.append(90-np.abs(angle))
        # cv2.imwrite(ALIGNED_GT_FOLDER + relativePatientPath + image_idx, img)

    print(angles)
    med_angle = statistics.median(angles[4:-4])
    print(med_angle)
    rot_matrix = cv2.getRotationMatrix2D(center=center, angle=med_angle, scale=1)
    for image_name in glob.glob(GT_FOLDER + relativePatientPath + "*"):
        image_idx = image_name.replace(GT_FOLDER + relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        orig_height, orig_width = img.shape[:2]
        rot_img = cv2.warpAffine(src=img, M=rot_matrix, dsize=(orig_width,orig_height))

        cv2.imwrite(ALIGNED_GT_FOLDER + relativePatientPath + image_idx, rot_img)


################################################################################
def readAndMirrorImages():
    patientFolders = glob.glob(ORIGINAL_FOLDER + "*/")

    for numFold, patientFolder in enumerate(patientFolders):  # for each patient
        relativePatientPath = patientFolder.replace(ORIGINAL_FOLDER, '')

        print("[INFO] - Analyzing {0}/{1}; patient folder: {2}...".format(numFold + 1, len(patientFolders), relativePatientPath))

        readAndMirrorGT(relativePatientPath)

        # readAndMirrorPM(relativePatientPath)

        # readAndMirror4DCTP(relativePatientPath,patientFolder)

        if MASK_NAME!="": readAndMirrorMask(relativePatientPath)


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
            if FLIP == 0: mirror_img = np.fliplr(img)
            elif FLIP == 1: mirror_img = np.flipud(img)
            else:
                print("problem with the FLIP flag {}".format(FLIP))
                return
            cv2.imwrite(MIRRORED_REGISTERED_FOLDER + relativePatientPath + slicefold + image_idx, mirror_img)


################################################################################
def readAndMirrorPM(relativePatientPath):
    if not os.path.isdir(MIRRORED_PM_FOLDER + relativePatientPath): os.mkdir(MIRRORED_PM_FOLDER + relativePatientPath)
    else:
        print("PMs for {} already exists, continue...".format(MIRRORED_PM_FOLDER + relativePatientPath))
        return

    if HASDAYFOLDER:  # SUS2020 dataset
        for dayfolder in glob.glob(PM_FOLDER + relativePatientPath + "*/"):
            # if the folder contains the correct number of subfolders
            if len(glob.glob(dayfolder + "*/")) >= 7:
                day_f = dayfolder.replace(PM_FOLDER+relativePatientPath, '')
                if not os.path.isdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f): os.mkdir(MIRRORED_PM_FOLDER+relativePatientPath+day_f)
                savePMImage(relativePatientPath, dayfolder,day_f=day_f)
    else:  # ISLES2018 dataset
        if len(glob.glob(PM_FOLDER + relativePatientPath + "*/"))>=6:
            savePMImage(relativePatientPath, PM_FOLDER + relativePatientPath)


################################################################################
def savePMImage(relativePatientPath, folder,day_f=""):
    # pmlist = ["CBF", "CBV", "TTP", "TMAX", "Tmax", "MIP", "MTT" ]
    pmlist = ["CBF", "CBV", "CT", "OT", "MTT", "Tmax"]
    for listpms in glob.glob(folder + "*/"):
        for pm in pmlist:
            if pm in listpms:
                if not os.path.isdir(MIRRORED_PM_FOLDER + relativePatientPath + day_f + pm + "/"):
                    os.mkdir(MIRRORED_PM_FOLDER + relativePatientPath + day_f + pm + "/")
                for image_name in glob.glob(listpms + "*"):
                    image_idx = image_name.replace(PM_FOLDER + relativePatientPath + day_f + pm + "/", '')
                    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                    if FLIP == 0: mirror_img = np.fliplr(img)
                    elif FLIP == 1: mirror_img = np.flipud(img)
                    else:
                        print("problem with the FLIP flag {}".format(FLIP))
                        return

                    if HASCOLORBAR:
                        mirror_img[:, 435:] = img[:, 435:]  # add the colorbar on the right in the mirrored image
                        mirror_img[:, :512 - 435] = 0  # remove the colorbar on the left in the mirrored image

                    cv2.imwrite(MIRRORED_PM_FOLDER + relativePatientPath + day_f + pm + "/" + image_idx, mirror_img)


################################################################################
def readAndMirrorGT(relativePatientPath):
    if not os.path.isdir(MIRRORED_GT_FOLDER + relativePatientPath): os.mkdir(MIRRORED_GT_FOLDER + relativePatientPath)
    else:
        print("GT for {} already exists, continue...".format(MIRRORED_GT_FOLDER + relativePatientPath))
        return

    for image_name in glob.glob(GT_FOLDER+relativePatientPath + "*"):
        image_idx = image_name.replace(GT_FOLDER+relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        if FLIP == 0: mirror_img = np.fliplr(img)
        elif FLIP == 1: mirror_img = np.flipud(img)
        else:
            print("problem with the FLIP flag {}".format(FLIP))
            return
        cv2.imwrite(MIRRORED_GT_FOLDER + relativePatientPath + image_idx, mirror_img)


def readAndMirrorMask(relativePatientPath):
    if not os.path.isdir(MIRRORED_MASK_FOLDER + relativePatientPath): os.mkdir(MIRRORED_MASK_FOLDER + relativePatientPath)
    else:
        print("Mask for {} already exists, continue...".format(MIRRORED_MASK_FOLDER + relativePatientPath))
        return
    for image_name in glob.glob(MASK_FOLDER + relativePatientPath + "*"):
        image_idx = image_name.replace(MASK_FOLDER + relativePatientPath, '')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        if FLIP == 0: mirror_img = np.fliplr(img)
        elif FLIP == 1: mirror_img = np.flipud(img)
        else:
            print("problem with the FLIP flag {}".format(FLIP))
            return
        cv2.imwrite(MIRRORED_MASK_FOLDER + relativePatientPath + image_idx, mirror_img)


################################################################################
# ## Main
################################################################################
if __name__ == '__main__':
    """
    Example usage for SUS2020 DS (& ISLES2018): 
    
    python augment_DS.py /home/prosjekt/PerfusionCT/StrokeSUS/ORIGINAL/ FINAL_Najm_v21-0.5/ Parametric_Maps/ GT_TIFF/ MASKS_v6/  -d -c
    
    python augment_DS.py /home/prosjekt/PerfusionCT/StrokeSUS/ISLES2018/Processed_TRAINING/ORIGINAL/ FINAL_v21-0.5/ Parametric_Maps/ Binary_Ground_Truth/ "" -f 0

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Set the root folder")
    parser.add_argument("ds_name", help="Set the dataset folder name")
    parser.add_argument("pm_name", help="Set the pms folder name")
    parser.add_argument("gt_name", help="Set the ground truth folder name")
    parser.add_argument("mask_name", help="Set the mask folder name")
    parser.add_argument("-f", "--flip", help="Set how tp flip the images for rotation", default=0, type=int, choices=[0,1])
    parser.add_argument("-d", "--dayfold", help="Flag for having a dayfolder in the parametric maps folder", action="store_true")
    parser.add_argument("-c", "--colorbar", help="Flag for removing the colorbar", action="store_true")

    args = parser.parse_args()

    ROOT_PATH = args.root
    DS_NAME = args.ds_name
    PM_NAME = args.pm_name
    GT_NAME = args.gt_name
    MASK_NAME = args.mask_name
    ORIGINAL_FOLDER = ROOT_PATH + DS_NAME

    ROOT_PATH = ROOT_PATH.replace("ORIGINAL","MIRRORED")
    if not os.path.isdir(ROOT_PATH): os.mkdir(ROOT_PATH)

    MIRRORED_REGISTERED_FOLDER = ROOT_PATH + "MIRRORED_" + DS_NAME
    PM_FOLDER = args.root + PM_NAME
    MIRRORED_PM_FOLDER = ROOT_PATH + "MIRRORED_" + PM_NAME
    GT_FOLDER = args.root + GT_NAME
    MIRRORED_GT_FOLDER = ROOT_PATH + "MIRRORED_" + GT_NAME
    if MASK_NAME!="":
        MASK_FOLDER = args.root + MASK_NAME
        MIRRORED_MASK_FOLDER = ROOT_PATH + "MIRRORED_" + MASK_NAME

    FLIP = args.flip
    HASDAYFOLDER = args.dayfold
    HASCOLORBAR = args.colorbar

    start = time.time()
    if not os.path.isdir(MIRRORED_REGISTERED_FOLDER): os.mkdir(MIRRORED_REGISTERED_FOLDER)
    if not os.path.isdir(MIRRORED_PM_FOLDER): os.mkdir(MIRRORED_PM_FOLDER)
    if not os.path.isdir(MIRRORED_GT_FOLDER): os.mkdir(MIRRORED_GT_FOLDER)
    if MASK_NAME!="" and not os.path.isdir(MIRRORED_MASK_FOLDER): os.mkdir(MIRRORED_MASK_FOLDER)

    print("Mirror the dataset...")
    readAndMirrorImages()
    end = time.time()
    print("Total time: {0}s".format(round(end-start, 3)))

    # print("Rotate the dataset to be vertically aligned.")
    # ROOT_PATH = ROOT_PATH.replace("MIRRORED", "ALIGNED")
    # if not os.path.isdir(ROOT_PATH): os.mkdir(ROOT_PATH)
    #
    # ALIGNED_REGISTERED_FOLDER = ROOT_PATH + DS_NAME
    # ALIGNED_PM_FOLDER = ROOT_PATH + PM_NAME
    # ALIGNED_GT_FOLDER = ROOT_PATH + GT_NAME
    # if MASK_NAME != "": ALIGNED_MASK_FOLDER = ROOT_PATH + MASK_NAME
    #
    # if not os.path.isdir(ALIGNED_REGISTERED_FOLDER): os.mkdir(ALIGNED_REGISTERED_FOLDER)
    # if not os.path.isdir(ALIGNED_PM_FOLDER): os.mkdir(ALIGNED_PM_FOLDER)
    # if not os.path.isdir(ALIGNED_GT_FOLDER): os.mkdir(ALIGNED_GT_FOLDER)
    # if MASK_NAME!="" and not os.path.isdir(ALIGNED_MASK_FOLDER): os.mkdir(ALIGNED_MASK_FOLDER)
    #
    # verticalAlignment()
    # print("Total time: {0}s".format(round(time.time()-end,3)))
