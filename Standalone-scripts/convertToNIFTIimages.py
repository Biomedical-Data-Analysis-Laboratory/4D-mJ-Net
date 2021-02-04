import os, cv2, glob, time
import numpy as np
import nibabel as nib
from tensorflow.keras.models import model_from_json

HOME = os.path.abspath("/home/stud/lucat/PhD_Project/Stroke_segmentation/")
experimentID = "EXP010";

config = dict()
config["main_folder"] = os.path.join(HOME, "PATIENTS/ISLES2018/")
config["data_folder"] = os.path.join(config["main_folder"], "TRAINING/")
config["save_folder"] = os.path.join(HOME, "SAVE/"+experimentID+"/IMAGES/mJNet_v2_DA_SGD_VAL5_64_256x256/")

config["test_data"]=False # if true uses all test data and do not count dice index
if config["test_data"]==True:
    config["data_folder"] =  os.path.join(config["main_folder"], "TESTING/")
    config["prediction_path"] = os.path.join(config["main_folder"], "prediction_test/")

config["train_data"]=True # if true uses all training data to evaluate dice
if config["train_data"]==True:
    config["prediction_path"] = os.path.join(config["main_folder"], "prediction_train/")

config["thres"] = 200 # # TODO: check the value

################################################################################
def getStringFromIndex(patient_index):
    p_id = str(patient_index)
    if len(p_id)==1: p_id = "0"+p_id

    return p_id

################################################################################
def convertImage(subject_folder, subsave_folder, caseID):
    if not os.path.exists(config["prediction_path"]): os.makedirs(config["prediction_path"])
    # load CT image to get SMIR ID, original size, header and afiine

    MTT_path = glob.glob(os.path.join(subject_folder, "*MTT.*"))[0] # get right ID for SMIR

    CT_path = glob.glob(os.path.join(subject_folder, "*CT.*"))[0]
    CT_filename = glob.glob(os.path.join(CT_path, "*CT.*.nii"))[0]
    CT = nib.load(CT_filename)

    prediction = np.zeros(CT.shape, np.int8)
    print("[INFO] - Prediction shape: ", prediction.shape)

    for imagename in np.sort(glob.glob(subsave_folder+"*test.png")): # sort the images !
        os.remove(imagename)

    for idx, imagename in enumerate(np.sort(glob.glob(subsave_folder+"*heatmap.png"))): # sort the images !
        image = cv2.imread(imagename, 0)
        label_map_data = np.zeros(np.array(image).shape, np.int8)
        label_map_data_tmp = np.zeros(np.array(image).shape, np.int8)
        label_map_data[image >= config["thres"]] = 1
        # label_map_data_tmp[image >= config["thres"]] = 255
        # cv2.imwrite(imagename.replace("heatmap","test"), label_map_data_tmp)

        prediction[:,:,idx] = label_map_data

    # print(CT.header)
    predNifti = nib.Nifti1Image(prediction, CT.affine, CT.header)
    prediction_path = os.path.join(config["prediction_path"], "SMIR.prediction_case" + caseID + "." + MTT_path.split(".")[-1] + ".nii")
    predNifti.to_filename(prediction_path)

################################################################################
def main():
    # choose which data use for evaluation
    if config["test_data"]==True:
        validation_indices = [i for i in range(1,63)]
    elif config["train_data"]==True:
        validation_indices = [i for i in range(1,95)]

    for i in validation_indices:
        caseID = getStringFromIndex(i)
        subject_folder = os.path.join(config["data_folder"],"case_" + str(i) + "/")
        print("[INFO] - Subject folder: " + subject_folder)
        subsave_folder = os.path.join(config["save_folder"],"PA" + caseID + "/")
        print("[INFO] - Saved Image folder: " + subsave_folder)

        convertImage(subject_folder, subsave_folder, caseID)

if __name__ == "__main__":
    main()
