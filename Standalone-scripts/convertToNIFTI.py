import os, cv2, glob, time, argparse
import numpy as np
import nibabel as nib


def convertStudyImages(patient):
    patient_id = patient.replace(ROOT_PATH,"")
    print("[INFO] - Patient folder: " + patient_id)
    savePatientFolder = os.path.join(SAVE_PATH, patient_id)
    if not os.path.exists(savePatientFolder): os.makedirs(savePatientFolder)

    CTP_study = np.zeros((512,512,len(glob.glob(patient + "*/")),30))
    for i_s, s in enumerate(np.sort(glob.glob(patient + "*/"))):
        for i_t, imagename in enumerate(np.sort(glob.glob(s + "*.tiff"))):
            img = cv2.imread(imagename)
            CTP_study[:,:,i_s,i_t] = img

    niftiStudy = nb.Nifti1Image(CTP_study)
    save_path = savePatientFolder+"study.nii.gz"
    nib.save(niftiStudy, save_path)

def main():
    # choose which data use for evaluation
    for patient in glob.glob(ROOT_PATH + "*/"):
        convertStudyImages(patient)


if __name__ == "__main__":
    """
        Example usage:
        python convertToNIFTI.py /home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/FINAL_Najm_v7/ /home/prosjekt/PerfusionCT/StrokeSUS/COMBINED/GT_TIFF/ /home/prosjekt/PerfusionCT/StrokeSUS/NIFTI/FINAL_Najm_v7/
        
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Set the root folder")
    parser.add_argument("gt_dir", help="Set the Ground Truth folder")
    parser.add_argument("save_dir", help="Set the save folder")

    args = parser.parse_args()

    ROOT_PATH = args.root_dir
    GT_PATH = args.gt_dir
    SAVE_PATH = args.save_dir

    main()
