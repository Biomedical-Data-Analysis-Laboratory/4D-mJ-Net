# Run the testing function, save the images ..
import cv2, time, glob, os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from tensorflow.keras import models
import warnings
import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K

from Model.constants import *
from Utils import general_utils, dataset_utils, sequence_utils, model_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


################################################################################
# Get the labeled image processed (= GT)
def get_check_img_processed(nn, p_id, idx):
    check_img_proc = np.zeros(shape=(get_img_width(), get_img_height()))
    if nn.labeled_img_folder != "":  # get the label image only if the path is set
        filename = None
        filename_1 = nn.labeled_img_folder + get_prefix_img() + p_id + os.path.sep + idx + SUFFIX_IMG
        filename_2 = nn.labeled_img_folder + get_prefix_img() + p_id + os.path.sep + idx + ".png"
        filename_3 = nn.labeled_img_folder + get_prefix_img() + p_id + os.path.sep + p_id + idx + SUFFIX_IMG
        for fname in [filename_1,filename_2,filename_3]:  # check if at least one path is correct
            if not os.path.exists(fname): continue
            filename = fname
        assert os.path.exists(filename), "[ERROR] - {0} does NOT exist".format(filename)

        check_img_proc = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        if len(check_img_proc.shape)==3: check_img_proc=cv2.cvtColor(check_img_proc, cv2.COLOR_BGR2GRAY)
        assert len(check_img_proc.shape) == 2, "The GT image shape should be 2."
    return check_img_proc


################################################################################
# Predict the model based on the input
def predict_from_model(nn, x_input, mcd=False):
    if mcd:  # MONTE CARLO DROPOUT
        n_samples = 100
        preds = np.zeros([get_m(),get_n(), get_n_classes()]) if is_TO_CATEG() else np.zeros([get_m(),get_n()])
        for n in tqdm.tqdm(range(n_samples)): preds += nn.model.predict(x=x_input, use_multiprocessing=nn.mp_in_nn)[0]
        #for n in range(n_samples): preds += K.eval(nn.model(x_input, training=True))[0] # add the various predictions
        preds /= n_samples  # divide them to get the mean prediction
        return [preds]
    else: return nn.model.predict(x=x_input, use_multiprocessing=nn.mp_in_nn)


################################################################################
# Save the intermediate layers
def save_intermediate_layers(model, x, idx, intermediate_act_path):
    if not os.path.isdir(intermediate_act_path + idx): os.mkdir(intermediate_act_path + idx)
    for layer in model.layers:
        if "conv" in layer.name or "leaky" in layer.name or "batch" in layer.name:
            layer_output = model.get_layer(layer.name).output
            intermediate_model = models.Model(inputs=model.input, outputs=layer_output)

            intermediate_pred = intermediate_model.predict(x, batch_size=1)
            print(layer.name, intermediate_pred.shape)
            if not os.path.isdir(intermediate_act_path + idx + os.path.sep + layer.name): os.mkdir(intermediate_act_path + idx + os.path.sep + layer.name)

            for img_index in range(0, intermediate_pred.shape[1]):
                plt.figure(figsize=(30, 30))
                xydim = int(np.ceil(np.sqrt(intermediate_pred.shape[4])))
                for c in range(0, intermediate_pred.shape[4]):
                    plt.subplot(xydim, xydim, c+1), plt.imshow(intermediate_pred[0, img_index, :, :, c], cmap='gray'), plt.axis('off')
                plt.savefig(intermediate_act_path + idx + os.path.sep + layer.name + os.path.sep + str(img_index) + ".png")


################################################################################
# Generate the images for the patient and save the images
def predict_and_save_img(nn, p_id):
    suffix = general_utils.get_suffix()  # es == "_4_16x16"

    suffix_filename = ".pkl"
    if nn.model_info["use_hickle"]: suffix_filename = ".hkl"
    filename_test = nn.ds_folder + DATASET_PREFIX + str(p_id) + suffix + suffix_filename

    if not os.path.exists(filename_test): return

    rel_patient_folder = get_prefix_img() + str(p_id) + os.path.sep
    rel_patient_fold_heatmap = rel_patient_folder + "HEATMAP" + os.path.sep
    rel_patient_fold_GT = rel_patient_folder + "GT" + os.path.sep
    rel_patient_folder_tmp = rel_patient_folder + "TMP" + os.path.sep
    patient_folder = nn.patients_folder + rel_patient_folder

    filename_save_img_folder = nn.save_img_folder + nn.experimentID + "__" + nn.get_nn_id() + suffix
    # create the related folders
    general_utils.create_dir(filename_save_img_folder)
    for subpath in [rel_patient_folder, rel_patient_fold_heatmap, rel_patient_fold_GT, rel_patient_folder_tmp]:
        general_utils.create_dir(filename_save_img_folder + os.path.sep + subpath)

    prefix = nn.experimentID + suffix_partial_weights + nn.get_nn_id() + suffix + os.path.sep
    subpatient_fold = prefix + rel_patient_folder
    patient_fold_heatmap = prefix + rel_patient_fold_heatmap
    patient_fold_GT = prefix + rel_patient_fold_GT
    patient_folder_tmp = prefix + rel_patient_folder_tmp

    # for all the slice folders in patientFolder
    for subfolder in glob.glob(patient_folder+"*"+os.path.sep):
        # Predict the images
        # s = time.time()
        if get_USE_PM(): predict_img_from_PMS(nn, subfolder, p_id, subpatient_fold, patient_fold_heatmap, patient_fold_GT, patient_folder_tmp, filename_test)
        else: predict_img(nn, subfolder, p_id, patient_folder, subpatient_fold, patient_fold_heatmap, patient_fold_GT, patient_folder_tmp, filename_test)
        # print("Time: {0}s".format(round(time.time() - s, 3)))


################################################################################
def predict_img(nn, subfolder, p_id, patient_folder, rel_patient_folder, rel_patient_fold_heatmap,  rel_patient_fold_GT,
                rel_patient_folder_tmp, filename_test):
    """
    Generate a SINGLE image for the patient and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - rel_patient_folder            : relative name of the patient folder
    - rel_patient_fold_heatmap      : relative name of the patient heatmap folder
    - rel_patient_fold_GT           : relative name of the patient gt folder
    - rel_patient_folder_tmp        : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """
    img_pred = np.zeros(shape=(get_img_width(), get_img_height()))
    categ_img = np.zeros(shape=(get_img_width(), get_img_height(), get_n_classes()))
    x, y = 0, 0
    X, coords = [], []
    idx = general_utils.get_str_from_idx(subfolder.replace(patient_folder, '').replace(os.path.sep, ""))  # image index
    # remove the old logs.
    logs_name = nn.save_img_folder + rel_patient_folder + idx + "_logs.txt"
    if os.path.isfile(logs_name): os.remove(logs_name)

    # if is_verbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))
    check_img_proc = get_check_img_processed(nn, str(p_id), idx)
    binary_mask = check_img_proc != get_pixel_values()[0]

    test_df = dataset_utils.read_pickle_or_hickle(filename_test, nn.model_info["use_hickle"])

    # Portion for the prediction of the image
    if is_3D() != "":
        assert os.path.exists(filename_test), "[ERROR] - File {} does NOT exist".format(filename_test)
        # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
        test_df = test_df[test_df.data_aug_idx==0]
        print(test_df.shape)
        test_df = test_df[test_df.sliceIndex==idx]
        print(test_df.shape)
        img_pred = generate_time_img_and_consensus(nn, test_df, rel_patient_folder_tmp, idx)
    else:  # usual behaviour
        while True:
            coords.append((x, y))
            # if we reach the end of the image, break the while loop.
            if x >= get_img_width() - get_m() and y >= get_img_height() - get_n(): break
            if get_m() == get_img_width() and get_n() == get_img_height(): break  # check for M == WIDTH & N == HEIGHT
            if y < get_img_height() - get_n(): y += get_n()  # going to the next slicing window
            else:
                if x < get_img_width():
                    y = 0
                    x += get_m()

        for i, coord in enumerate(coords):
            row = test_df[test_df.x_y == coord]
            row = row[row.sliceIndex == idx]
            row = row[row.data_aug_idx == 0]
            # Control that the analyzed row is == 1
            assert len(row) == 1, "The length of the row to analyze should be 1."
            X = model_utils.get_correct_X_for_input_model(ds_seq=nn.test_sequence, current_folder=subfolder, row=row,
                                                          batch_idx=i, batch_len=len(coords), X=X)

        img_pred, categ_img = generate_2D_img(nn, X, coords, img_pred, categ_img, binary_mask, idx)

    if nn.model_info["save_images"]:  # save the image
        save_img(nn, rel_patient_folder, idx, img_pred, categ_img, rel_patient_fold_heatmap, rel_patient_fold_GT, rel_patient_folder_tmp, check_img_proc)


################################################################################
# Function for predicting a brain slice based on the parametric maps
def predict_img_from_PMS(nn, subfolder, p_id, rel_patient_folder, rel_patient_fold_heatmap, rel_patient_fold_GT,
                         rel_patient_folder_tmp, filename_test):
    """
    Generate ALL the images for the patient using the PMs and save it.

    Input parameters:
    - nn                            : NeuralNetwork class
    - subfolder                     : Name of the slice subfolder
    - p_id                          : Patient ID
    - patientFolder                 : folder of the patient
    - rel_patient_folder            : relative name of the patient folder
    - rel_patient_fold_heatmap      : relative name of the patient heatmap folder
    - rel_patient_fold_GT           : relative name of the patient gt folder
    - rel_patient_folder_tmp        : relative name of the patient tmp folder
    - filename_test                 : Name of the test pandas dataframe
    """

    # if the patient folder contains the correct number of subfolders
    # ATTENTION: careful here...
    n_fold = 7 if not is_ISLES2018() else 5

    if len(glob.glob(subfolder+"*"+os.path.sep))>=n_fold:
        pmcheckfold = os.path.sep+"CBF"+os.path.sep
        for idx in glob.glob(subfolder+pmcheckfold+"*"):
            idx = general_utils.get_str_from_idx(idx.replace(subfolder, '').replace(pmcheckfold, ""))  # image index
            if is_ISLES2018(): idx = idx.replace(".tiff", "")
            else: idx = idx.replace(".png","")
            # remove the old logs.
            logsName = nn.save_img_folder + rel_patient_folder + idx + "_logs.txt"
            if os.path.isfile(logsName): os.remove(logsName)

            # if getVerbose(): print("[INFO] - Analyzing Patient {0}, image {1}.".format(p_id, idx))

            checkImageProcessed = get_check_img_processed(nn, str(p_id), idx)

            assert os.path.exists(filename_test), "[ERROR] - File {0} does NOT exist".format(filename_test)

            # get the pandas dataframe
            test_df = dataset_utils.read_pickle_or_hickle(filename_test, nn.model_info["use_hickle"])
            # get only the rows with data_aug_idx==0 (no rotation or any data augmentation)
            test_df = test_df[test_df.data_aug_idx == 0]
            test_df = test_df[test_df.sliceIndex == idx]

            img_pred, categ_img = generate_img_from_PMS(nn, test_df, idx)

            if nn.model_info["save_images"]:  # save the image
                save_img(nn, rel_patient_folder, idx, img_pred, categ_img, rel_patient_fold_heatmap,
                         rel_patient_fold_GT, rel_patient_folder_tmp, checkImageProcessed)


################################################################################
# Util function to save image
def save_img(nn, rel_patient_folder, idx, img_pred, categ_img, rel_patient_fold_heatmap, rel_patient_folder_GT,
             rel_patient_folder_tmp, check_img_processed):
    # save the image predicted in the specific folder
    cv2.imwrite(nn.save_img_folder + rel_patient_folder + idx + ".png", img_pred)
    # create and save the HEATMAP only if we are using softmax activation
    if is_TO_CATEG() and nn.labeled_img_folder!= "":
        p_idx, c_idx = 2,3
        if get_n_classes() ==3:p_idx, c_idx = 1, 2
        elif get_n_classes() ==2: c_idx = 1

        checkImageProcessed_rgb = cv2.cvtColor(check_img_processed, cv2.COLOR_GRAY2RGB)
        if get_n_classes() >= 3:
            heatmap_img_p = cv2.convertScaleAbs(categ_img[:, :, p_idx] * 255)
            heatmap_img_p = cv2.applyColorMap(heatmap_img_p, cv2.COLORMAP_JET)
            blend_p = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_p, 0.5, 0.0)
            cv2.imwrite(nn.save_img_folder + rel_patient_fold_heatmap + idx + "_heatmap_penumbra.png", blend_p)
        heatmap_img_c = cv2.convertScaleAbs(categ_img[:, :, c_idx] * 255)
        heatmap_img_c = cv2.applyColorMap(heatmap_img_c, cv2.COLORMAP_JET)
        blend_c = cv2.addWeighted(checkImageProcessed_rgb, 0.5, heatmap_img_c, 0.5, 0.0)
        cv2.imwrite(nn.save_img_folder + rel_patient_fold_heatmap + idx + "_heatmap_core.png", blend_c)

    # Save the ground truth and the contours
    if is_3D() == "" and nn.labeled_img_folder!= "":
        # save the GT
        cv2.imwrite(nn.save_img_folder + rel_patient_folder_GT + idx + SUFFIX_IMG, check_img_processed)

        img_pred = cv2.cvtColor(np.uint8(img_pred), cv2.COLOR_GRAY2RGB)  # back to rgb
        if get_n_classes() >= 3:
            _, penumbra_mask = cv2.threshold(check_img_processed, 85, get_pixel_values()[-2], cv2.THRESH_BINARY)
            penumbra_cnt, _ = cv2.findContours(penumbra_mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            img_pred = cv2.drawContours(img_pred, penumbra_cnt, -1, (255, 0, 0), 2)
        _, core_mask = cv2.threshold(check_img_processed, get_pixel_values()[-2], get_pixel_values()[-1], cv2.THRESH_BINARY)
        core_cnt, _ = cv2.findContours(core_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_pred = cv2.drawContours(img_pred, core_cnt, -1, (0, 0, 255), 2)
        # save the GT image with predicted contours
        # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, penumbra_area, 0.5, 0.0)
        # checkImageProcessed = cv2.addWeighted(checkImageProcessed, 1, core_area, 0.5, 0.0)
        cv2.imwrite(nn.save_img_folder + rel_patient_folder_tmp + idx + ".png", img_pred)


################################################################################
# Helpful function that return the 2D image from the pixel and the starting coordinates
def generate_2D_img(nn, pixels, coords, img_pred, categ_img, binary_mask, idx):
    """
    Generate a 2D image from the test_df

    Input parameters:
    - nn                    : NeuralNetwork class
    - pixels                : pixel in a numpy array
    - coords                : (x,y) coordinates
    - img_pred              : the predicted image
    - categ_img             : the categorical image predicted
    - binary_mask           : the binary mask containing the skull

    Return:
    - img_pred              : the predicted image
    - categ_img             : the categorical image predicted
    """
    # swp_orig contain only the prediction for the last step
    swp_orig_arr = predict_from_model(nn, pixels, mcd=nn.model_info["MONTE_CARLO_DROPOUT"])
    for i,swp_orig in enumerate(swp_orig_arr):
        x, y = coords[i]
        if nn.model_info["save_images"] and is_TO_CATEG(): categ_img[x:x+get_m(),y:y+get_n()] = swp_orig
        # convert the categorical into a single array for removing some uncertain predictions
        if is_TO_CATEG(): slice_window_pred = (np.argmax(swp_orig,axis=-1) * 255) / (get_n_classes() - 1)
        else: slice_window_pred = swp_orig * 255
        # save the predicted images
        if nn.model_info["save_images"]:
            if not has_limited_columns() and get_n_classes() >2:
                # Remove the parts already classified by the model
                binary_mask = np.array(binary_mask, dtype=np.float)
                # force all the predictions to be inside the binary mask defined by the GT
                slice_window_pred *= binary_mask[x:x + get_m(), y:y + get_n()]

                overlapping_pred = np.array(slice_window_pred > 0, dtype=np.float)
                overlapping_pred *= 85.
                binary_mask *= 85.  # multiply the binary mask for the brain pixel value
                # add the brain to the prediction window
                slice_window_pred += (binary_mask[x:x + get_m(), y:y + get_n()] - overlapping_pred)
            img_pred[x:x + get_m(), y:y + get_n()] = slice_window_pred

    if nn.model_info["save_activation_filter"]:
        if idx == "03" or idx == "04": save_intermediate_layers(nn.model, pixels, idx, intermediate_act_path=nn.intermediate_activation_folder)

    return img_pred, categ_img


################################################################################
def generate_time_img_and_consensus(nn, test_df, rel_patient_folder_tmp, idx):
    """
    Generate the image from the 3D sequence of time index (create also these images) with a consensus

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - rel_patient_folder_tmp    : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - img_pred                  : the predicted image
    - categ_img             : the categorical image predicted
    """

    img_pred = np.zeros(shape=(get_img_width(), get_img_height()))
    categ_img = np.zeros(shape=(get_img_width(), get_img_height(), get_n_classes()))
    check_img_processed = np.zeros(shape=(get_img_width(), get_img_height()))
    array_time_idx_img = dict()

    for test_row in test_df.itertuples():  # for every rows of the same image
        if str(test_row.timeIndex) not in array_time_idx_img.keys(): array_time_idx_img[str(test_row.timeIndex)] = np.zeros(shape=(get_img_width(),
                                                                                                                                   get_img_height()), dtype=np.uint8)
        if is_3D() == "": test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2], 1)
        else: test_row.pixels = test_row.pixels.reshape(1, test_row.pixels.shape[0], test_row.pixels.shape[1], test_row.pixels.shape[2])
        array_time_idx_img[str(test_row.timeIndex)], categ_img = generate_2D_img(nn, test_row.pixels, [test_row.x_y],
                                                                                 array_time_idx_img[str(test_row.timeIndex)],
                                                                                 categ_img, check_img_processed, idx)

    if nn.model_info["save_images"]:  # remove one class from the ground truth
        if get_n_classes() ==3: check_img_processed[check_img_processed == 85] = get_pixel_values()[0]
        cv2.imwrite(nn.save_img_folder + rel_patient_folder_tmp + "orig_" + idx + SUFFIX_IMG, check_img_processed)

        for tidx in array_time_idx_img.keys():
            curr_image = array_time_idx_img[tidx]
            # save the images
            cv2.imwrite(nn.save_img_folder + rel_patient_folder_tmp + idx + "_" + general_utils.get_str_from_idx(tidx) + SUFFIX_IMG, curr_image)
            # add the predicted image in the imagePredicted for consensus
            img_pred += curr_image

        img_pred /= len(array_time_idx_img.keys())

    return img_pred, categ_img


################################################################################
# Function to predict an image starting from the parametric maps
def generate_img_from_PMS(nn, test_df, idx):
    """
    Generate a 2D image from the test_df using the parametric maps

    Input:
    - nn                        : NeuralNetwork class
    - test_df                   : pandas dataframe for testing
    - checkImageProcessed       : the labeled image (Ground truth img)
    - rel_patient_folder_tmp    : tmp folder for the patient
    - idx                       : image index (slice)

    Return:
    - img_pred                  : the predicted image
    - categ_img                 : the categorical image predicted
    """

    img_pred = np.zeros(shape=(get_img_width(), get_img_height()))
    categ_img = np.zeros(shape=(get_img_width(), get_img_height(), get_n_classes()))
    start_x, start_y = 0, 0
    constants ={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                "weights": get_class_weights_const()[0], "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(), "USE_PM":get_USE_PM(),
                "LIST_PMS":get_list_PMS(), "IMAGE_HEIGHT":get_img_height(), "IMAGE_WIDTH": get_img_width()}

    while True:
        row_to_analyze = test_df[test_df.x_y == (start_x, start_y)]

        assert len(row_to_analyze) == 1, "The length of the row to analyze should be 1."

        binary_mask = np.zeros(shape=(get_m(), get_n()))
        pms = dict()
        for pm_name in get_list_PMS():
            filename = row_to_analyze[pm_name].iloc[0]
            pm = cv2.imread(filename, nn.input_img_flag)
            pms[pm_name] = general_utils.get_slice_window(pm, start_x, start_y, constants, train=True, remove_colorbar=True)
            # add the mask of the pixels that are > 0 only if it's the MIP image
            if pm_name=="MIP":
                if nn.params["convertImgToGray"]: binary_mask += pms[pm_name] > 0
                else: binary_mask += (cv2.cvtColor(pms[pm_name], cv2.COLOR_RGB2GRAY) > 0)
            pms[pm_name] = np.array(pms[pm_name])
            pms[pm_name] = pms[pm_name].reshape((1,) + pms[pm_name].shape)
            if nn.params["concatenate_input"] and nn.params["inflate_network"]: pms[pm_name] = pms[pm_name].reshape((1,)+pms[pm_name].shape)
            elif nn.params["concatenate_input"]: pms[pm_name] = pms[pm_name].reshape(pms[pm_name].shape + (1,))

        X = []
        if "cbf" in nn.multi_input.keys() and nn.multi_input["cbf"] == 1: X.append(pms["CBF"])
        if "cbv" in nn.multi_input.keys() and nn.multi_input["cbv"] == 1: X.append(pms["CBV"])
        if "ttp" in nn.multi_input.keys() and nn.multi_input["ttp"] == 1: X.append(pms["TTP"])
        if "mtt" in nn.multi_input.keys() and nn.multi_input["mtt"] == 1: X.append(pms["MTT"])
        if "tmax" in nn.multi_input.keys() and nn.multi_input["tmax"] == 1: X.append(pms["TMAX"])
        if "mip" in nn.multi_input.keys() and nn.multi_input["mip"] == 1: X.append(pms["MIP"])
        if "nihss" in nn.multi_input.keys() and nn.multi_input["nihss"] == 1: X.append(np.array([int(row_to_analyze["NIHSS"].iloc[0])]) if row_to_analyze["NIHSS"].iloc[0] != "-" else np.array([0]))
        if "age" in nn.multi_input.keys() and nn.multi_input["age"] == 1: X.append(np.array([int(row_to_analyze["age"].iloc[0])]))
        if "gender" in nn.multi_input.keys() and nn.multi_input["gender"] == 1: X.append(np.array([int(row_to_analyze["gender"].iloc[0])]))

        # slicingWindowPredicted contain only the prediction for the last step
        img_pred, categ_img = generate_2D_img(nn, X, [(start_x, start_y)], img_pred, categ_img, binary_mask, idx)

        # if we reach the end of the image, break the while loop.
        if start_x>=get_img_width()-get_m() and start_y>=get_img_height()-get_n(): break
        # check for M == WIDTH & N == HEIGHT
        if get_m()==get_img_width() and get_n()==get_img_height(): break
        # going to the next slicingWindow
        if start_y<(get_img_height()-get_n()): start_y+=get_n()
        else:
            if start_x < get_img_width():
                start_y = 0
                start_x += get_m()

    return img_pred, categ_img


################################################################################
# Test the model with the selected patient
def evaluate_model(nn, p_id, is_already_saved, i):
    suffix = general_utils.get_suffix()

    if is_already_saved:
        suffix_filename = ".pkl"
        if nn.model_info["use_hickle"]: suffix_filename = ".hkl"
        filename_train = nn.ds_folder + DATASET_PREFIX + str(p_id) + suffix + suffix_filename
        assert os.path.exists(filename_train), "The filename for the DF {} does not exist".format(filename_train)
        # Read the dataframe given the filename of it
        nn.train_df = dataset_utils.read_pickle_or_hickle(filename_train, nn.model_info["use_hickle"])

        nn.dataset = dataset_utils.get_test_ds(nn.dataset, nn.train_df, p_id, nn.use_sequence, nn.mp_in_nn)
        if not nn.use_sequence: nn.dataset["test"]["labels"] = dataset_utils.get_labels_from_idx(train_df=nn.train_df,dataset=nn.dataset["test"],modelname=nn.name,flag="test")
        nn.compile_model()  # compile the model and then evaluate it

    sample_weights = nn.get_sample_weights("test")
    if nn.use_sequence:
        nn.test_sequence = sequence_utils.ds_sequence(
            dataframe=nn.train_df,
            indices=nn.dataset["test"]["indices"],
            x_label=nn.x_label,
            y_label=nn.y_label,
            multi_input=nn.multi_input,
            batch_size=nn.batch_size,
            params=nn.params,
            back_perc=100,
            is3dot5DModel=nn.is3dot5DModel,
            is4DModel=nn.is4DModel,
            inputImgFlag=nn.input_img_flag,
            supervised=nn.model_info["supervised"],
            patients_folder=nn.patients_folder,
            labeled_img_folder=nn.labeled_img_folder,
            constants={"M": get_m(), "N": get_m(), "NUMBER_OF_IMAGE_PER_SECTION": getNUMBER_OF_IMAGE_PER_SECTION(),
                       "TIME_LAST": is_timelast(), "N_CLASSES": get_n_classes(), "PIXELVALUES": get_pixel_values(),
                       "weights": get_class_weights_const()[0], "TO_CATEG": is_TO_CATEG(), "isISLES": is_ISLES2018(),
                       "USE_PM": get_USE_PM(), "LIST_PMS": get_list_PMS(),
                       "IMAGE_HEIGHT": get_img_height(), "IMAGE_WIDTH": get_img_width()},
            name=nn.name,
            flagtype="test",
            loss=nn.loss["name"])
    else:
        testing = nn.model.evaluate(
            x=nn.dataset["test"]["data"],
            y=nn.dataset["test"]["labels"],
            callbacks=nn.callbacks,
            sample_weight=sample_weights,
            verbose=is_verbose(),
            batch_size=nn.batch_size,
            use_multiprocessing=nn.mp_in_nn)

        general_utils.print_sep("-", 50)
        for metric_name in nn.train.history: print("TRAIN %s: %.2f%%" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
        for index, val in enumerate(testing): print("TEST %s: %.2f%%" % (nn.model.metrics_names[index], round(val,6)*100))
        general_utils.print_sep("-", 50)

        with open(general_utils.get_dir_path(nn.save_text_folder) + nn.get_nn_id() + suffix + ".txt", "a+") as text_file:
            for metric_name in nn.train.history: text_file.write("TRAIN %s: %.2f%% \n" % (metric_name, round(float(nn.train.history[metric_name][-1]), 6)*100))
            for index, val in enumerate(testing): text_file.write("TEST %s: %.2f%% \n" % (nn.model.metrics_names[index], round(val,6)*100))
            text_file.write("----------------------------------------------------- \n")
