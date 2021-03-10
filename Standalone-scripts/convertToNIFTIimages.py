import os, cv2, glob, time
import numpy as np
import nibabel as nib
# from tensorflow.keras.models import model_from_json

HOME = os.path.abspath("/home/stud/lucat/PhD_Project/Stroke_segmentation/")
experimentID = "EXP200.1"

config = dict()
config["main_folder"] = os.path.join(HOME, "PATIENTS/ISLES2018/")

config["train_data"]=True  # if true uses all training data to evaluate dice
config["test_data"]=True  # if true uses all test data and do not count dice index
flag_pred = "prediction_train_"
flag_save = "IMAGES/"
flag_data = "TRAINING/"

if config["test_data"]:
    flag_pred = "prediction_test_"
    flag_save = "TEST_IMAGES/"
    flag_data = "TESTING/"

config["data_folder"] = os.path.join(config["main_folder"], flag_data)
config["save_folder"] = os.path.join(HOME, "SAVE/"+experimentID+"/"+flag_save+experimentID+"__PMs_segmentation_NOBatch_DA_ADAM_VAL20_SOFTMAX_64_256x256/")
config["prediction_path"] = os.path.join(config["main_folder"], flag_pred+experimentID+"/")

config["thres"] = 200  # # TODO: check the value


################################################################################
def getStringFromIndex(patient_index):
    p_id = str(patient_index)
    if len(p_id)==1: p_id = "0"+p_id
    return p_id


################################################################################
def convertImage(subject_folder, subsave_folder, caseID):
    if not os.path.exists(config["prediction_path"]): os.makedirs(config["prediction_path"])
    # load CT image to get SMIR ID, original size, header and affine

    MTT_path = glob.glob(os.path.join(subject_folder, "*MTT.*"))[0]  # get right ID for SMIR
    MTT_filename = glob.glob(os.path.join(MTT_path, "*MTT.*.nii"))[0]
    MTT = nib.load(MTT_filename)
    CT_path = glob.glob(os.path.join(subject_folder, "*CT.*"))[0]
    CT_filename = glob.glob(os.path.join(CT_path, "*CT.*.nii"))[0]
    CT = nib.load(CT_filename)

    prediction = np.zeros(CT.shape, np.ushort)

    for idx, imagename in enumerate(np.sort(glob.glob(subsave_folder+"*.png"))):
        image = cv2.imread(imagename,0)
        label_map_data = np.zeros(np.array(image).shape, np.ushort)
        label_map_data[image >= config["thres"]] = 1

        prediction[:,:,idx] = label_map_data

    # CT.set_data_dtype(MTT.get_data_dtype())
    predNifti = nib.Nifti1Image(prediction, MTT.affine, header=MTT.header)
    prediction_path = os.path.join(config["prediction_path"], "SMIR.prediction_case" + caseID + "." + MTT_filename.split(".")[-2] + ".nii")
    # predNifti.to_filename(prediction_path)
    nib.save(predNifti, prediction_path)


################################################################################
def main():
    # choose which data use for evaluation
    if config["test_data"]: validation_indices = [i for i in range(1,63)]
    elif config["train_data"]: validation_indices = [i for i in range(1,95)]
    for i in validation_indices:
        caseID = getStringFromIndex(i)
        subject_folder = os.path.join(config["data_folder"],"case_" + str(i) + "/")
        print("[INFO] - Subject folder: " + subject_folder)
        subsave_folder = os.path.join(config["save_folder"],"PA" + caseID + "/")
        print("[INFO] - Saved Image folder: " + subsave_folder)

        convertImage(subject_folder, subsave_folder, caseID)


if __name__ == "__main__":
    main()
